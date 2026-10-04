from transformers import MarianMTModel, MarianTokenizer

texts = [
    'నాకు రెండు రోజుల నుంచి కడుపు నొప్పిగా ఉంది',
    'నాకు కడుపు నొప్పి లేదు',
    'నాకు రేపు కాలేజీకి వెళ్లాలి',
    'నాకు నీళ్లు కావాలి',
]

for model_id in ['Helsinki-NLP/opus-mt-dra-en', 'Helsinki-NLP/opus-mt-tel-en']:
    print('MODEL:', model_id)
    try:
        tok = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id)
        for t in texts:
            enc = tok(t, return_tensors='pt', padding=True, truncation=True)
            out = model.generate(**enc, max_length=128)
            print('SRC:', t)
            print('DST:', tok.decode(out[0], skip_special_tokens=True))
            print('---')
    except Exception as e:
        print('ERROR:', type(e).__name__, e)
    print('====')
