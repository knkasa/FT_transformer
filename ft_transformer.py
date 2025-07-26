import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import pdb

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        
        self.norm1 = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
            ])
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):

        attn_out = self.attn(self.norm1(x), self.norm1(x))
        x = x + self.dropout1(attn_out, training=training)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out, training=training)
        
        return x
    
class FTTransformer(tf.keras.Model):
    def __init__(self, num_cat_features, cat_cardinalities, num_cont_features,
                 d_model=32, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1, output_dim=1):
        super().__init__()

        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.d_model = d_model

        self.cat_embeddings = [
            layers.Embedding(input_dim=card, output_dim=d_model)
            for card in cat_cardinalities
            ]

        self.cont_proj = layers.Dense(d_model)
        self.cls_token = self.add_weight("cls_token", shape=(1, 1, d_model), initializer="random_normal")

        self.encoder_layers = [
            TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
            ]

        # Output layer. Make it regression or classification.
        self.head = layers.Dense(output_dim)

    def call(self, inputs, training=False):
        pdb.set_trace()
        cat_inputs, cont_inputs = inputs
        batch_size = tf.shape(cat_inputs)[0]

        cat_tokens = [embed(cat_inputs[:, i]) for i, embed in enumerate(self.cat_embeddings)]
        cat_tokens = tf.stack(cat_tokens, axis=1)  # shape: (batch_size, num_cat_features, d_model)

        cont_inputs_reshaped = tf.expand_dims(cont_inputs, axis=-1)  # (batch, num_cont_features, 1)
        cont_tokens = self.cont_proj(cont_inputs_reshaped)           # (batch, num_cont_features, d_model)
        tokens = tf.concat([cat_tokens, cont_tokens], axis=1)        # (batch, total_tokens, d_model)

        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])     # (batch, 1, d_model)
        tokens = tf.concat([cls_tokens, tokens], axis=1)             # (batch, 1 + total_tokens, d_model)

        for layer in self.encoder_layers:
            tokens = layer(tokens, training=training)

        cls_output = tokens[:, 0, :]
        return self.head(cls_output)

#========================================================================

num_samples = 1000
num_cat_features = 3
num_cont_features = 2
batch_size = 32
cat_cardinalities = [2, 5, 10]  # vocab sizes for categorical columns. 2 for binary

# Categorical features
cat_data = np.column_stack([
    np.random.randint(0, card, size=num_samples) for card in cat_cardinalities
    ])

# Continuous features: random floats
cont_data = np.random.randn(num_samples, num_cont_features)

target = (
    cat_data[:, 0] * 0.5
    + cat_data[:, 1] * 0.3
    + cont_data[:, 0] * 2.0
    - cont_data[:, 1] * 1.5
    + np.random.randn(num_samples) * 0.5  # noise
    )

cat_train, cat_val, cont_train, cont_val, y_train, y_val = train_test_split(
    cat_data, cont_data, target, test_size=0.2, random_state=42
    )

scaler = StandardScaler()
cont_train = scaler.fit_transform(cont_train)
cont_val = scaler.transform(cont_val)

#---------- Training -------------

# instead of doing this, you could use list [cat_train, cont_train]
train_dataset = tf.data.Dataset.from_tensor_slices(((cat_train, cont_train), y_train))
train_dataset = train_dataset.shuffle(512).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(((cat_val, cont_val), y_val))
val_dataset = val_dataset.batch(batch_size)

model = FTTransformer(
    num_cat_features=num_cat_features,
    cat_cardinalities=cat_cardinalities,
    num_cont_features=num_cont_features,
    output_dim=1
    )

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
