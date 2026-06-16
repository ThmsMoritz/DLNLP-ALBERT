from transformers import AlbertConfig, AlbertForSequenceClassification

config = AlbertConfig(
    num_hidden_layers=12,
    num_hidden_groups=12,
)

model = AlbertForSequenceClassification(config)

groups = model.albert.encoder.albert_layer_groups

print("Number of groups:", len(groups))

for i, group in enumerate(groups):
    print(
        f"Group {i}:",
        len(group.albert_layers)
    )