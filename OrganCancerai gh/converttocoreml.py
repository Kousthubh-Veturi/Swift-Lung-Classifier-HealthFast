import tensorflow as tf
import coremltools as ct


filename = "lungimagesets.h5"
model = tf.keras.models.load_model(filename)
coreml_path = filename.rsplit('.', 1)[0] + '.mlmodel'
print(coreml_path)

coreml_model = ct.convert(model, inputs=[ct.ImageType(scale=1/255.0)])
coreml_model.save("lungimagesets.mlpackage")

