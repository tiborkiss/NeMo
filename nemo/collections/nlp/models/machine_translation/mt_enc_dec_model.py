# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as pt_data
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator, TransformerEmbedding, TopKSequenceGenerator
from nemo.core.classes.common import typecheck
from nemo.utils import logging

__all__ = ['MTEncDecModel']


class MTEncDecModel(EncDecNLPModel):
    """
    Encoder-decoder machine translation model.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None):

        self.setup_enc_dec_tokenizers(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        self.encoder = instantiate(cfg.encoder)
        self.encoder_embedding = TransformerEmbedding(
            vocab_size=self.encoder_vocab_size,
            hidden_size=cfg.encoder.hidden_size,
            max_sequence_length=cfg.encoder_embedding.max_sequence_length,
            num_token_types=cfg.encoder_embedding.num_token_types,
            embedding_dropout=cfg.encoder_embedding.embedding_dropout,
            learn_positional_encodings=cfg.encoder_embedding.learn_positional_encodings,
        )
        self.decoder = instantiate(cfg.decoder)
        self.decoder_embedding = TransformerEmbedding(
            vocab_size=self.decoder_vocab_size,
            hidden_size=cfg.decoder.hidden_size,
            max_sequence_length=cfg.decoder_embedding.max_sequence_length,
            num_token_types=cfg.decoder_embedding.num_token_types,
            embedding_dropout=cfg.decoder_embedding.embedding_dropout,
            learn_positional_encodings=cfg.decoder_embedding.learn_positional_encodings,
        )

        self.log_softmax = TokenClassifier(
            hidden_size=cfg.decoder.hidden_size,
            num_classes=self.decoder_vocab_size,
            activation=cfg.head.activation,
            log_softmax=cfg.head.log_softmax,
            dropout=cfg.head.dropout,
            use_transformer_init=cfg.head.use_transformer_init,
        )

        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.decoder_embedding,
            decoder=self.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=cfg.decoder_embedding.max_sequence_length,
            beam_size=cfg.beam_size,
            bos=self.decoder_tokenizer.bos_id,
            pad=self.decoder_tokenizer.pad_id,
            eos=self.decoder_tokenizer.eos_id,
            len_pen=cfg.len_pen,
            max_delta_length=cfg.max_generation_delta,
        )

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.decoder_embedding.token_embedding.weight

        # TODO: encoder and decoder with different hidden size?
        std_init_range = 1 / cfg.encoder.hidden_size ** 0.5
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.decoder_tokenizer.pad_id, label_smoothing=cfg.label_smoothing
        )

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

        # self.training_perplexity = Perplexity(dist_sync_on_step=True)
        # self.eval_perplexity = Perplexity(compute_on_step=False)

    def filter_predicted_ids(self, ids):
        ids[ids >= self.decoder_tokenizer.vocab_size] = self.decoder_tokenizer.unk_id
        return ids

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        torch.nn.Module.forward method.
        Args:
            src: source ids
            src_mask: src mask (mask padding)
            tgt: target ids
            tgt_mask: target mask

        Returns:

        """
        src_embeddings = self.encoder_embedding(input_ids=src)
        src_hiddens = self.encoder(src_embeddings, src_mask)
        if tgt is not None:
            tgt_embeddings = self.decoder_embedding(input_ids=tgt)
            tgt_hiddens = self.decoder(tgt_embeddings, tgt_mask, src_hiddens, src_mask)
            log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        else:
            log_probs = None
        beam_results = None
        if not self.training:
            beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
            beam_results = self.filter_predicted_ids(beam_results)
        return log_probs, beam_results

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels, _ = batch
        log_probs, _ = self(src_ids, src_mask, tgt_ids, tgt_mask)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        # training_perplexity = self.training_perplexity(logits=log_probs)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
            # "train_ppl": training_perplexity,
        }
        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids = batch
        log_probs, beam_results = self(src_ids, src_mask, tgt_ids, tgt_mask)
        eval_loss = self.loss_fn(log_probs=log_probs, labels=labels).cpu().numpy()
        # self.eval_perplexity(logits=log_probs)
        translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
        np_tgt = tgt_ids.cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()
        tensorboard_logs = {f'{mode}_loss': eval_loss}
        return {
            f'{mode}_loss': eval_loss,
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
            'log': tensorboard_logs,
        }

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @rank_zero_only
    def log_param_stats(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
                self.trainer.logger.experiment.add_scalars(
                    name,
                    {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
                    global_step=self.global_step,
                )

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx, 'val')

    def eval_epoch_end(self, outputs, mode):
        counts = np.array([x['num_non_pad_tokens'] for x in outputs])
        eval_loss = np.sum(np.array([x[f'{mode}_loss'] for x in outputs]) * counts) / counts.sum()
        # eval_perplexity = self.eval_perplexity.compute()
        translations = list(itertools.chain(*[x['translations'] for x in outputs]))
        ground_truths = list(itertools.chain(*[x['ground_truths'] for x in outputs]))
        # __TODO__ add target language so detokenizer can be lang specific.
        detokenizer = MosesDetokenizer()
        translations = [detokenizer.detokenize(sent.split()) for sent in translations]
        ground_truths = [detokenizer.detokenize(sent.split()) for sent in ground_truths]
        assert len(translations) == len(ground_truths)
        sacre_bleu = corpus_bleu(translations, [ground_truths], tokenize="13a")
        dataset_name = "Validation" if mode == 'val' else "Test"
        logging.info(f"\n\n\n\n{dataset_name} set size: {len(translations)}")
        logging.info(f"{dataset_name} Sacre BLEU = {sacre_bleu.score}")
        logging.info(f"{dataset_name} TRANSLATION EXAMPLES:".upper())
        for i in range(0, 3):
            ind = random.randint(0, len(translations) - 1)
            logging.info("    " + '\u0332'.join(f"EXAMPLE {i}:"))
            logging.info(f"    Prediction:   {translations[ind]}")
            logging.info(f"    Ground Truth: {ground_truths[ind]}")

        ans = {f"{mode}_loss": eval_loss, f"{mode}_sacreBLEU": sacre_bleu.score}  # , f"{mode}_ppl": eval_perplexity}
        ans['log'] = dict(ans)
        return ans

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.log_dict(self.eval_epoch_end(outputs, 'val'))

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        if cfg.get("load_from_cached_dataset", False):
            logging.info('Loading from cached dataset %s' % (cfg.src_file_name))
            if cfg.src_file_name != cfg.tgt_file_name:
                raise ValueError("src must be equal to target for cached dataset")
            dataset = pickle.load(open(cfg.src_file_name, 'rb'))
            dataset.reverse_lang_direction = cfg.get("reverse_lang_direction", False)
        else:
            dataset = TranslationDataset(
                dataset_src=str(Path(cfg.src_file_name).expanduser()),
                dataset_tgt=str(Path(cfg.tgt_file_name).expanduser()),
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                cache_ids=cfg.get("cache_ids", False),
                cache_data_per_node=cfg.get("cache_data_per_node", False),
                use_cache=cfg.get("use_cache", False),
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
            )
            dataset.batchify(self.encoder_tokenizer, self.decoder_tokenizer)
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
    
    def replace_beam_with_sampling(self, topk=500):
        self.beam_search = TopKSequenceGenerator(
            embedding=self.decoder_embedding,
            decoder=self.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.beam_search.max_seq_length,
            beam_size=topk,
            bos=self.decoder_tokenizer.bos_id,
            pad=self.decoder_tokenizer.pad_id,
            eos=self.decoder_tokenizer.eos_id,
        )

    @torch.no_grad()
    def translate(self, text: List[str], target_lang: str = 'en') -> List[str]:
        """
        Translates list of sentences from source language to target language.
        Should be regular text, this method performs its own tokenization/de-tokenization
        Args:
            text: list of strings to translate

        Returns:
            list of translated strings
        """
        mode = self.training
        detokenizer = MosesDetokenizer(lang=target_lang)
        try:
            self.eval()
            res = []
            for txt in text:
                ids = self.encoder_tokenizer.text_to_ids(txt)
                ids = [self.encoder_tokenizer.bos_id] + ids + [self.encoder_tokenizer.eos_id]
                src = torch.Tensor(ids).long().to(self._device).unsqueeze(0)
                src_mask = torch.ones_like(src)
                src_embeddings = self.encoder_embedding(input_ids=src)
                src_hiddens = self.encoder(src_embeddings, src_mask)
                beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
                beam_results = self.filter_predicted_ids(beam_results)
                translation_ids = beam_results.cpu()[0].numpy()
                translation = self.decoder_tokenizer.ids_to_text(translation_ids)
                res.append(detokenizer.detokenize(translation.split()))
        finally:
            self.train(mode=mode)
        return res

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
