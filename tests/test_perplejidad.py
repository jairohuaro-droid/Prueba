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


class TestEvaluador(unittest.TestCase):
    """Usa un modelo diminuto; se omite si no hay transformers o red."""

    @classmethod
    def setUpClass(cls):
        try:
            from detector.perplejidad import Evaluador
            modelo = "sshleifer/tiny-gpt2"
            cls.evaluador = Evaluador(observador=modelo, ejecutor=modelo, dispositivo="cpu")
        except Exception as e:  # dependencias o descarga no disponibles
            raise unittest.SkipTest(f"Modelo no disponible: {e}")

    def test_senales(self):
        s = self.evaluador.analizar(TEXTO)
        self.assertGreater(s["perplejidad"], 1)
        self.assertTrue(0 <= s["fraccion_top10"] <= 1)
        self.assertIsNotNone(s["binoculars"])
        self.assertIsNotNone(s["burstiness"])


if __name__ == "__main__":
    unittest.main()
