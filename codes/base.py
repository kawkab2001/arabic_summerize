# =====================================================================
# تلخيص عربي عام - النسخة الهرمية مع اختيار الهدف الديناميكي
#
# ما الذي تغيّر؟
# 1. النموذج لا يزال هرمياً عاماً (بدون audience embeddings).
# 2. أثناء تجهيز البيانات: يقرأ عمود "Type" ويختار عمود الملخص المطابق له.
#    (Type=Kids -> Summary_Kids, Type=Adults -> Summary_Adult, etc.)
# 3. التقييم النهائي يقارن التوليد مع الملخص المرجعي الصحيح بناءً على Type.
# =====================================================================
#BASE CODE
import os
import re
import json
import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import evaluate
import nltk
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    set_seed,
    TrainerCallback,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

# ======================
# الإعدادات
# ======================
MODEL_NAME = "Osame1/AraT5-msa-small-finetuned-xlsum-ar"
DATA_FILE  = "/content/drive/MyDrive/summmery_new/finall.csv"
OUTPUT_DIR = "/content/drive/MyDrive/summmery_new/base"


COL_ID   = "ID"
COL_TEXT = "original"
COL_TYPE = "Type"

# خريطة أعمدة الملخصات
AUDIENCE_COLS = {
    "kids":   "Summary_Kids",
    "adult":  "Summary_Adult",
    "expert": "Summary_Expert",
}

# خريطة توحيد قيم عمود Type
TYPE_MAP = {
    "Kids": "kids", "Adults": "adult", "Experts": "expert",
    "kids": "kids", "adult": "adult", "expert": "expert",
    "أطفال": "kids", "بالغون": "adult", "خبراء": "expert",
}
 
MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 192
BATCH_SIZE        = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS            = 12
 
LR                = 1e-4
SEED              = 42
BETA_SALIENCE     = 0.1  # تعزيز التعلم الهرمي للكلمات المفتاحية
WARMUP_RATIO      = 0.15
 
DEFAULT_GENERATION_CONFIG = {
    "num_beams": 4, "length_penalty": 1.0, "no_repeat_ngram_size": 3,
    "early_stopping": True, "do_sample": False,
    "min_length": 10, "repetition_penalty": 1.0,
}
 
set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ الجهاز المستخدم: {device}")
_use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"🔢 وضع الدقة: {'bf16' if _use_bf16 else 'float32'}")
 
_TOK_KW = "processing_class" if "processing_class" in inspect.signature(
    Seq2SeqTrainer.__init__).parameters else "tokenizer"
 
# ======================
# تحميل البيانات واختيار الهدف ديناميكياً
# ======================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"الملف '{path}' غير موجود.")
    df = pd.read_csv(path)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["ID"] = range(1, len(df) + 1)
    df.columns = [c.strip() for c in df.columns]
 
    needed = [COL_ID, COL_TEXT, COL_TYPE] + list(AUDIENCE_COLS.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"أعمدة مفقودة: {missing}\nالموجودة: {list(df.columns)}")
 
    df = df[needed].dropna()
    for c in needed[1:]:
        df[c] = df[c].astype(str).str.strip()
    
    # توحيد قيم Type
    df["mapped_type"] = df[COL_TYPE].map(TYPE_MAP)
    bad_types = df["mapped_type"].isna()
    if bad_types.any():
        print(f"⚠️ تم حذف {bad_types.sum()} صفوف بسبب قيم Type غير معروفة: {df[COL_TYPE][bad_types].unique()}")
        df = df[~bad_types].reset_index(drop=True)

    df = df.drop_duplicates(subset=COL_ID).reset_index(drop=True)
    df = df[df["mapped_type"].apply(lambda x: len(str(x).split()) >= 1)].reset_index(drop=True)
 
    print(f"\n📊 عدد النصوص الصالحة: {len(df)}")
    print("📊 توزيع نوع النص الأصلي (Type):")
    print(df["mapped_type"].value_counts().to_string())
    return df
 
 
wide_df = load_data(DATA_FILE)
 
_rng = np.random.RandomState(SEED)
_ids = wide_df[COL_ID].values.copy()
_rng.shuffle(_ids)
_n_test = max(1, int(0.05 * len(_ids)))
_n_val  = max(1, int(0.05 * len(_ids)))
test_ids  = set(_ids[:_n_test])
val_ids   = set(_ids[_n_test:_n_test + _n_val])
train_ids = set(_ids[_n_test + _n_val:])
 
train_df = wide_df[wide_df[COL_ID].isin(train_ids)].reset_index(drop=True)
val_df   = wide_df[wide_df[COL_ID].isin(val_ids)].reset_index(drop=True)
test_df  = wide_df[wide_df[COL_ID].isin(test_ids)].reset_index(drop=True)
print(f"نصوص: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
 
 
def to_dataset(frame):
    """
    هنا يحدث السحر: نقرأ mapped_type ونختار عمود الملخص المناسب له كهدف للتدريب.
    """
    data = {"id": [], "text": [], "summary": [], "true_type": []}
    for _, r in frame.iterrows():
        t_type = r["mapped_type"]
        target_col = AUDIENCE_COLS[t_type]  # <-- الاختيار الديناميكي
        
        data["id"].append(str(r[COL_ID]))
        data["text"].append(r[COL_TEXT])
        data["summary"].append(r[target_col]) # نضع الملخص الصحيح في عمود واحد موحد
        data["true_type"].append(t_type)      # نحتفظ بالنوع للتقييم لاحقاً
        
    return Dataset.from_dict(data)
 
train_ds_raw = to_dataset(train_df)
val_ds_raw   = to_dataset(val_df)
test_ds_raw  = to_dataset(test_df)
 
# ======================
# المعالج النصي
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
 
SPECIAL_IDS = set(tokenizer.all_special_ids)
 
def preprocess_seq2seq(examples):
    texts = examples["text"]
    targets = examples["summary"]

    enc_inputs = tokenizer(texts, max_length=MAX_INPUT_LENGTH, truncation=True)
    enc_targets = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)

    labels_list, salience_list = [], []

    for inp_ids, target_ids in zip(enc_inputs["input_ids"], enc_targets["input_ids"]):
        target_set = set(target_ids) - SPECIAL_IDS
        salience = [1 if tok in target_set else 0 for tok in inp_ids]
        labels_list.append(target_ids)
        salience_list.append(salience)

    return {
        "input_ids": enc_inputs["input_ids"],
        "attention_mask": enc_inputs["attention_mask"],
        "labels": labels_list,
        "salience_labels": salience_list,
    }
 
def tokenize_ds(ds):
    return ds.map(preprocess_seq2seq, batched=True, remove_columns=ds.column_names)
 
train_ds = tokenize_ds(train_ds_raw)
eval_ds  = tokenize_ds(val_ds_raw)
 
print(f"✅ بيانات التدريب: train={len(train_ds)} | val={len(eval_ds)}")
 
 
# ======================
# النموذج الهرمي (Hierarchical T5) - بدون تعقيدات الجمهور
# ======================
class HierarchicalT5(nn.Module):
    main_input_name = "input_ids"
 
    def __init__(self, name, adapter_dim=256):
        super().__init__()
        self.base = AutoModelForSeq2SeqLM.from_pretrained(name)
        d = self.base.config.d_model
 
        self.adapter = nn.Sequential(nn.Linear(d, adapter_dim), nn.ReLU(), nn.Linear(adapter_dim, d))
        self.sent_proj = nn.Linear(d, d)
        self.salience_head = nn.Linear(d, 1)
 
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        nn.init.normal_(self.sent_proj.weight, std=0.02)
        nn.init.zeros_(self.sent_proj.bias)
 
        self.config = self.base.config
        self._custom_modules = ["adapter", "sent_proj", "salience_head"]
 
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base.prepare_inputs_for_generation(*args, **kwargs)
 
    def _reorder_cache(self, *args, **kwargs):
        return self.base._reorder_cache(*args, **kwargs)
 
    @property
    def generation_config(self):
        return self.base.generation_config
 
    @generation_config.setter
    def generation_config(self, value):
        self.base.generation_config = value
 
    def sentence_context(self, hidden, attention_mask):
        B, T, D = hidden.shape
        chunk = 20
        ctx = torch.zeros_like(hidden)
        for i in range(0, T, chunk):
            h = hidden[:, i:i + chunk, :]
            m = attention_mask[:, i:i + chunk].unsqueeze(-1).to(hidden.dtype)
            pooled = (h * m).sum(1) / (m.sum(1) + 1e-8)
            ctx[:, i:i + chunk, :] = self.sent_proj(pooled).unsqueeze(1)
        return ctx
 
    def encode(self, input_ids, attention_mask):
        enc_out = self.base.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        H = enc_out.last_hidden_state
        H = H + self.sentence_context(H, attention_mask)
        H = H + self.adapter(H)
        return H
 
    def forward(self, input_ids=None, attention_mask=None, labels=None, salience_labels=None, **kwargs):
        H = self.encode(input_ids, attention_mask)
        salience_logits = self.salience_head(H).squeeze(-1)
 
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=H)
        decoder_outputs = self.base(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs_obj,
            labels=labels,
            return_dict=True,
        )
        
        total_loss = decoder_outputs.loss if decoder_outputs.loss is not None else torch.tensor(0.0, device=H.device)
 
        if salience_labels is not None and BETA_SALIENCE > 0:
            bce = F.binary_cross_entropy_with_logits(salience_logits.float(), salience_labels.float(), reduction="none")
            m = attention_mask.float()
            sal_loss = (bce * m).sum() / m.sum().clamp(min=1.0)
            if torch.isfinite(sal_loss):
                total_loss = total_loss + BETA_SALIENCE * sal_loss
 
        if not torch.isfinite(total_loss):
            total_loss = torch.tensor(0.0, device=H.device, requires_grad=True)
 
        decoder_outputs.loss = total_loss
        decoder_outputs.salience_logits = salience_logits
        return decoder_outputs
 
    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        for key in ["salience_labels", "labels", "decoder_input_ids", "decoder_attention_mask"]:
            kwargs.pop(key, None)
        kwargs.setdefault("use_cache", True)
        kwargs.setdefault("decoder_start_token_id", tokenizer.pad_token_id)
        kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
 
        H = self.encode(input_ids, attention_mask)
        if kwargs.get("use_cache", True) and "past_key_values" not in kwargs:
            try:
                from transformers import EncoderDecoderCache, DynamicCache
                kwargs["past_key_values"] = EncoderDecoderCache(DynamicCache(), DynamicCache())
            except Exception:
                kwargs["use_cache"] = False
 
        return self.base.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=H),
            attention_mask=attention_mask,
            **kwargs
        )
 
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.base.save_pretrained(save_directory, safe_serialization=False)
        custom_state = {name: {k: v.cpu() for k, v in getattr(self, name).state_dict().items()} for name in self._custom_modules}
        custom_state["config"] = {"adapter_dim": self.adapter[0].out_features}
        torch.save(custom_state, os.path.join(save_directory, "custom_modules.pt"))
        self.config.save_pretrained(save_directory)
 
    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        custom_path = os.path.join(save_directory, "custom_modules.pt")
        custom_state = torch.load(custom_path, map_location="cpu", weights_only=False)
        cfg = custom_state.get("config", {})
        model = cls(save_directory, adapter_dim=cfg.get("adapter_dim", 256), **kwargs)
        for mod_name in model._custom_modules:
            if mod_name in custom_state:
                getattr(model, mod_name).load_state_dict(custom_state[mod_name])
        return model
 
    def unfreeze_top_encoder_layers(self, num_layers=4):
        total_layers = self.base.config.num_layers
        for name, param in self.base.encoder.named_parameters():
            parts = name.split(".")
            if parts[0] == "block" and len(parts) > 1 and parts[1].isdigit():
                if int(parts[1]) >= total_layers - num_layers:
                    param.requires_grad = True
            elif "final_layer_norm" in name:
                param.requires_grad = True
 
 
# ======================
# جامع البيانات
# ======================
class HierarchicalCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model=None, label_pad_token_id=-100):
        super().__init__(tokenizer=tokenizer, model=model, label_pad_token_id=label_pad_token_id)

    def __call__(self, features):
        salience = [f.pop("salience_labels", None) for f in features]
        batch = super().__call__(features)
        if salience[0] is not None:
            max_in_len = len(batch["input_ids"][0])
            padded_salience = [s + [0] * (max_in_len - len(s)) for s in salience]
            batch["salience_labels"] = torch.tensor(padded_salience, dtype=torch.float)
        return batch
 
seq_collator = HierarchicalCollator(tokenizer, model=None, label_pad_token_id=-100)
 
 
# ======================
# المقاييس
# ======================
bertscore_metric = evaluate.load("bertscore")
chrf_metric      = evaluate.load("chrf")
bleu_metric      = evaluate.load("bleu")
meteor_metric    = evaluate.load("meteor")
rouge_metric     = evaluate.load("rouge")
 
def ws_tok(x): return x.split()
 
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple): preds = preds[0]
    vocab_size = tokenizer.vocab_size
    preds  = np.clip(np.where(preds != -100, preds, tokenizer.pad_token_id).astype(np.int64), 0, vocab_size - 1)
    labels = np.clip(np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int64), 0, vocab_size - 1)
    
    dp = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    dl = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
    
    return {
        "bertscore_f1": float(np.mean(bertscore_metric.compute(predictions=dp, references=dl, lang="ar")["f1"])),
        "chrf":   chrf_metric.compute(predictions=dp, references=dl)["score"],
        "bleu":   bleu_metric.compute(predictions=dp, references=[[l] for l in dl])["bleu"],
        "meteor": meteor_metric.compute(predictions=dp, references=dl)["meteor"],
        "rouge1": rouge_metric.compute(predictions=dp, references=dl, tokenizer=ws_tok)["rouge1"],
        "rouge2": rouge_metric.compute(predictions=dp, references=dl, tokenizer=ws_tok)["rouge2"],
        "rougeL": rouge_metric.compute(predictions=dp, references=dl, tokenizer=ws_tok)["rougeL"],
        "gen_len": float(np.mean([len(x.split()) for x in dp])),
    }
 
 
# ======================
# التوقف المبكر والمدرب
# ======================
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.0, metric="eval_loss", greater_is_better=False):
        self.patience, self.min_delta, self.metric, self.gib = patience, min_delta, metric, greater_is_better
        self.best, self.wait, self.best_state = None, 0, None
 
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        val = (metrics or {}).get(self.metric)
        if val is None: return
        improved = (self.best is None or (val > self.best + self.min_delta if self.gib else val < self.best - self.min_delta))
        if improved:
            self.best, self.wait = val, 0
            if model is not None: self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True
                print(f"\n⚠️ التوقف المبكر: {self.metric} لم يتحسن لمدة {self.patience} تقييمات")
 
    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None and self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"✅ تم استرجاع أفضل نموذج ({self.metric} = {self.best:.4f})")
 
 
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tok is not None: tok.save_pretrained(output_dir)
 
    def create_optimizer(self):
        if self.optimizer is not None: return self.optimizer
        custom_ids = {id(p) for mod_name in self.model._custom_modules for p in getattr(self.model, mod_name).parameters()}
        groups = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad: continue
            key = (id(p) in custom_ids, p.ndim == 1 or name.endswith(".bias") or "layer_norm" in name)
            groups.setdefault(key, []).append(p)
        
        param_groups = [{"params": params, "lr": self.args.learning_rate, "weight_decay": 0.0 if no_decay else self.args.weight_decay} 
                        for (is_custom, no_decay), params in groups.items()]
        opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        opt_kwargs.pop("lr", None)
        self.optimizer = opt_cls(param_groups, **opt_kwargs)
        return self.optimizer
 
 
def _common_args(**over):
    import inspect as _inspect
    _ta_params = set(_inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    warmup_kwargs = {"warmup_ratio": WARMUP_RATIO} if "warmup_ratio" in _ta_params else {"warmup_steps": 100}
    base = dict(lr_scheduler_type="linear", weight_decay=0.01, per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, logging_steps=50,
                bf16=_use_bf16, fp16=False, report_to="none", seed=SEED, remove_unused_columns=False,
                load_best_model_at_end=False, max_grad_norm=1.0, **warmup_kwargs)
    base.update(over)
    return Seq2SeqTrainingArguments(**{k: v for k, v in base.items() if k in _ta_params})
 
 
def make_trainer(model, args, train_ds, eval_ds, collator, callbacks, compute_metrics_fn=None):
    return CustomSeq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                                data_collator=collator, compute_metrics=compute_metrics_fn,
                                callbacks=callbacks, **{_TOK_KW: tokenizer})
 
 
# ======================
# دوال التوليد والتدريب
# ======================
@torch.no_grad()
def summarize(text, model, tokenizer, device="cuda"):
    model.eval(); model.to(device)
    enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
    gen = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, **DEFAULT_GENERATION_CONFIG,
                         decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
 
 
def train_model(model, train_ds, eval_ds, epochs=12):
    print("\n" + "=" * 60)
    print("🔄 التدريب: نموذج هرمي عام مع أهداف ديناميكية حسب Type")
    print("=" * 60)
    
    for p in model.parameters(): p.requires_grad = False
    for m_name in model._custom_modules:
        for p in getattr(model, m_name).parameters(): p.requires_grad = True

    total_enc_layers = model.base.config.num_layers
    layers_to_unfreeze = max(2, total_enc_layers // 2)
    
    for name, param in model.base.encoder.named_parameters():
        parts = name.split(".")
        if parts[0] == "block" and len(parts) > 1 and parts[1].isdigit():
            if int(parts[1]) >= total_enc_layers - layers_to_unfreeze: param.requires_grad = True
        elif "final_layer_norm" in name: param.requires_grad = True

    for param in model.base.decoder.parameters(): param.requires_grad = True
    if hasattr(model.base, 'lm_head'):
        for param in model.base.lm_head.parameters(): param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ بارامترات قابلة للتدريب: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    args = _common_args(output_dir=OUTPUT_DIR, learning_rate=LR, num_train_epochs=epochs,
                        predict_with_generate=True, generation_max_length=MAX_TARGET_LENGTH,
                        generation_config=GenerationConfig(**DEFAULT_GENERATION_CONFIG), label_names=["labels"])
    
    tr = make_trainer(model, args, train_ds, eval_ds, collator=seq_collator,
                      callbacks=[EarlyStoppingCallback(patience=4, metric="eval_rougeL", greater_is_better=True)],
                      compute_metrics_fn=compute_metrics)
    tr.train()
    return tr
 
 
# ======================
# بدء التنفيذ
# ======================
print("\n🏁 بدء تدريب النموذج الهرمي الديناميكي...")
model = HierarchicalT5(MODEL_NAME)
model.config.decoder_start_token_id = tokenizer.pad_token_id
 
trainer = train_model(model, train_ds, eval_ds, epochs=EPOCHS)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
 
# عرض أمثلة
print("\n" + "=" * 80)
print("📝 أمثلة على التلخيص (النموذج سيحاول مطابقة أسلوب النص الأصلي):")
sample_texts = [
    "كان يا ما كان في قديم الزمان أرنب صغير يحب اللعب في الغابة كل يوم مع أصدقائه.",
    "أعلنت الحكومة عن حزمة إصلاحات اقتصادية تستهدف تخفيض معدلات البطالة خلال العامين القادمين.",
]
for text in sample_texts:
    print(f"\n📄 النص: {text}")
    print(f"📝 الملخص: {summarize(text, model, tokenizer, device)}")
    print("-" * 80)
 
 
# التقييم النهائي (مطابقة ديناميكية مع المرجع الصحيح)
print("\n" + "=" * 60)
print("📈 التقييم النهائي على مجموعة الاختبار (مطابقة مع الملخص الصحيح حسب Type)")
print("=" * 60)
 
preds, refs = [], []
for _, row in test_df.iterrows():
    text = row[COL_TEXT]
    true_type = row["mapped_type"]
    target_col = AUDIENCE_COLS[true_type] # <-- الاختيار الديناميكي للمرجع
    
    preds.append(summarize(text, model, tokenizer, device))
    refs.append(str(row[target_col]).strip())
 
r = rouge_metric.compute(predictions=preds, references=refs, tokenizer=ws_tok)
b = bertscore_metric.compute(predictions=preds, references=refs, lang="ar")
c = chrf_metric.compute(predictions=preds, references=refs)
 
report = {
    "rouge1": r["rouge1"], "rouge2": r["rouge2"], "rougeL": r["rougeL"],
    "bertscore_f1": float(np.mean(b["f1"])), "chrf": c["score"],
    "pred_words": float(np.mean([len(p.split()) for p in preds])),
    "ref_words":  float(np.mean([len(x.split()) for x in refs])),
}
print(pd.DataFrame([report]).T.round(3).to_string())
 
out_df = test_df[[COL_ID, COL_TEXT, "mapped_type"]].copy()
out_df["pred_summary"] = preds
# إضافة المرجع الصحيح للتقييم
out_df["ref_summary"] = refs
out_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False, encoding="utf-8-sig")
 
metrics = trainer.evaluate()
if metrics.get("eval_loss") is not None:
    metrics["perplexity"] = math.exp(min(metrics["eval_loss"], 50))
for k, v in sorted(metrics.items()):
    print(f"{k:35s}: {v:.4f}" if isinstance(v, float) else f"{k:35s}: {v}")
 
print("\n" + "✨" * 30)
print("🎉 اكتمل التدريب والتقييم!")
print(f"💾 كل شيء محفوظ في: {OUTPUT_DIR}")