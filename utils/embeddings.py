import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import tf_sentencepiece, sentencepiece
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def generate_features(module, texts):
	module_url = {
		"USE": "https://tfhub.dev/google/universal-sentence-encoder-large/3",
		"XLING": "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1",
		"USE_multilingual": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
	}[module]
	embed = hub.load(module_url)
	# texts = tf.constant(texts)
	features = embed(texts).numpy()
	return features
