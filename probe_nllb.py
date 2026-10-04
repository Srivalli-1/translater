from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

texts = [
    'నాకు రెండు రోజుల నుంచి కడుపు నొప్పిగా ఉంది',
    'నాకు కడుపు నొప్పి లేదు',
    'నాకు రేపు కాలేజీకి వెళ్లాలి',
    'నాకు నీళ్లు కావాలి',
]

model_id = 'facebook/nllb-200-distilled-600M'
print('MODEL:', model_id)

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

for t in texts:
    inputs = tok(t, return_tensors='pt')
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tok.lang_code_to_id['eng_Latn'],
        max_length=128,
        num_beams=4,
    )
    out = tok.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print('SRC:', t)
    print('DST:', out)
    print('---')
