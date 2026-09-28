import tempfile
import unittest

from detector.perplejidad import spans_oraciones

TEXTO = (
    "Ayer se me rompió la bici. Otra vez. "
    "Iba bajando por la calle de mi tía y la cadena hizo un ruido rarísimo. "
    "Tuve que volver caminando."
)


class TestSpans(unittest.TestCase):
    def test_spans_oraciones(self):
        partes = [TEXTO[a:b] for a, b in spans_oraciones(TEXTO)]
        self.assertEqual(partes[0], "Ayer se me rompió la bici.")
        self.assertEqual(partes[1], "Otra vez.")
        self.assertEqual(len(partes), 4)


def crear_modelo_diminuto(directorio, vocab=300):
    """GPT-2 de 2 capas con pesos aleatorios y un tokenizador BPE entrenado
    sobre TEXTO. Sirve para probar el código sin descargar nada."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([TEXTO] * 20, trainers.BpeTrainer(
        vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(directorio)
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, n_embd=32,
                        bos_token_id=None, eos_token_id=None)
    GPT2LMHeadModel(config).save_pretrained(directorio)
    return tok.get_vocab_size()


class TestEvaluador(unittest.TestCase):
    """Usa un modelo diminuto local; se omite si no hay torch/transformers."""

    @classmethod
    def setUpClass(cls):
        try:
            from detector.perplejidad import Evaluador
            cls.directorio = tempfile.TemporaryDirectory()
            cls.vocab = crear_modelo_diminuto(cls.directorio.name)
            cls.evaluador = Evaluador(observador=cls.directorio.name,
                                      ejecutor=cls.directorio.name, dispositivo="cpu")
        except ImportError as e:
            raise unittest.SkipTest(f"Dependencias no instaladas: {e}")

    @classmethod
    def tearDownClass(cls):
        cls.directorio.cleanup()

    def test_senales(self):
        s = self.evaluador.analizar(TEXTO)
        # Con pesos aleatorios la perplejidad ronda el tamaño del vocabulario.
        self.assertTrue(1 < s["perplejidad"] < 2 * self.vocab)
        self.assertTrue(0 <= s["fraccion_top10"] <= 1)
        self.assertIsNotNone(s["burstiness"])
        # Mismo modelo como observador y ejecutor → Binoculars ≈ 1.
        self.assertAlmostEqual(s["binoculars"], 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
