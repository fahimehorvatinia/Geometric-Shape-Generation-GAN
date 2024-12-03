import numpy as np

def train_gan(generator, discriminator, gan, data, epochs, batch_size):
    for epoch in range(epochs):
        # Train Discriminator
        real_data = data[np.random.randint(0, data.shape[0], batch_size)]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_data = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"Epoch {epoch}: D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")
