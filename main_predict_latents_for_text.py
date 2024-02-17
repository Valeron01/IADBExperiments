import clip

model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
model = model.eval()
text = clip.tokenize([""]).to("cpu")
print(text)
text_features = model.encode_text_to_features(text)
print(text_features.shape)
