from deepsky.gan import generator_model, discriminator_model, encoder_model


def test_equal_filter_sizes():
    gen = generator_model(min_data_width=2, min_conv_filters=32)
    disc = discriminator_model(min_data_width=2, min_conv_filters=32)
    enc = encoder_model(min_data_width=2, min_conv_filters=32)
    print(gen.summary())
    print(disc.summary())
    print(enc.summary())
    assert gen.layers[0].output_shape[-1] == disc.layers[-3].output_shape[-1]
    assert gen.layers[0].output_shape[-1] == enc.layers[-3].output_shape[-1]


if __name__ == "__main__":
    test_equal_filter_sizes()