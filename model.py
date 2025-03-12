import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = 400


STYLE_LAYERS = [ 
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)
]

def load_image(image_file):
    """Loads, resizes, and normalizes an image."""
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0  
        return img
    except Exception as e:
        print(f"Error loading image {image_file}: {e}")
        return None

def gram_matrix(A):

    A = tf.reshape(A, (-1, A.shape[-1]))  
    return tf.matmul(tf.transpose(A), A)

def compute_content_cost(content_output, generated_output):
   
    return tf.reduce_mean(tf.square(content_output[-1] - generated_output[-1]))

def compute_layer_style_cost(a_S, a_G):
   
    GS, GG = gram_matrix(a_S), gram_matrix(a_G)
    return tf.reduce_mean(tf.square(GG - GS))

def compute_style_cost(style_outputs, generated_outputs):
   
    J_style = 0
    for i, (layer_name, weight) in enumerate(STYLE_LAYERS):
        J_style += weight * compute_layer_style_cost(style_outputs[i], generated_outputs[i])
    return J_style

def get_vgg_model():
   
    vgg = tf.keras.applications.VGG19(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    vgg.trainable = False
    return vgg

def get_layer_outputs(vgg, layer_names):
  
    outputs = [vgg.get_layer(name).output for name, _ in layer_names]
    return tf.keras.Model([vgg.input], outputs)

def tensor_to_image(tensor):
    
    tensor = np.clip(tensor.numpy() * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor[0])

def train_step(generated_image, optimizer, vgg_model_outputs, a_C, a_S):
   
    with tf.GradientTape() as tape:
        a_G = vgg_model_outputs(generated_image)
        J_content = compute_content_cost(a_C, a_G)
        J_style = compute_style_cost(a_S, a_G)
        J_total = 10 * J_content + 40 * J_style

    grads = tape.gradient(J_total, generated_image)
    optimizer.apply_gradients([(grads, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

    return J_total

def run(content_image, style_image, epochs=10):

    tf.random.set_seed(272)

    vgg = get_vgg_model()
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + [('block5_conv4', 1)])

    content_image = tf.convert_to_tensor(content_image[None, ...], dtype=tf.float32)
    style_image = tf.convert_to_tensor(style_image[None, ...], dtype=tf.float32)

    a_C = vgg_model_outputs(content_image)
    a_S = vgg_model_outputs(style_image)

    generated_image = tf.Variable(content_image, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

    for i in range(epochs):
        loss = train_step(generated_image, optimizer, vgg_model_outputs, a_C, a_S)
        if i % 2 == 0:
            print(f"Epoch {i}: Loss = {loss.numpy()}")

    return tensor_to_image(generated_image)

def reset_images():

    folder = "images/"
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    content_img = load_image("images/Louvre_Museum.jpg")
    style_img = load_image("images/painting-impressionist-style.jpg")

    if content_img is None or style_img is None:
        print("Error loading images. Exiting.")
        exit(1)

    generated_img = run(content_img, style_img, epochs=10)

    plt.imshow(generated_img)
    plt.axis("off")
    plt.show()

    generated_img.save("images/generated_image.jpg")
    print("Generated image saved.")

    reset_images()
