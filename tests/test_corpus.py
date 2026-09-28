import unittest

from corpus.construir import en_espanol, recortar
from corpus.evaluar import auroc, dividir, umbral_para_fpr


class TestConstruir(unittest.TestCase):
    def test_recortar_respeta_oraciones(self):
        texto = "Una dos tres. Cuatro cinco seis. Siete ocho nueve."
        self.assertEqual(recortar(texto, 6), "Una dos tres. Cuatro cinco seis.")
        # Nunca devuelve vacío aunque la primera oración sea larga.
        self.assertEqual(recortar(texto, 1), "Una dos tres.")

    def test_en_espanol(self):
        self.assertTrue(en_espanol("El análisis de la política de salud en los hospitales del país"))
        self.assertFalse(en_espanol("This research aims at explaining the donors' trust"))


class TestEvaluar(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(auroc([3, 4], [1, 2]), 1.0)
        self.assertEqual(auroc([1, 2], [3, 4]), 0.0)
        self.assertEqual(auroc([1], [1]), 0.5)

    def test_umbral_para_fpr(self):
        humanos = [i / 100 for i in range(100)]
        u = umbral_para_fpr(humanos, 0.05)
        self.assertLessEqual(sum(h >= u for h in humanos), 5)

    def test_dividir_mantiene_pares(self):
        filas = []
        for i in range(10):
            filas.append({"id": f"t{i}", "grupo": "tesis", "gemelo_de": "", "etiqueta": "humano"})
            filas.append({"id": f"ia{i}", "grupo": "gemelos", "gemelo_de": f"t{i}", "etiqueta": "ia"})
        filas.append({"id": "c", "grupo": "clasicos", "gemelo_de": "", "etiqueta": "humano"})
        a, b = dividir(filas)
        self.assertEqual(len(a) + len(b), 20)
        llaves_a = {f["gemelo_de"] or f["id"] for f in a}
        self.assertFalse(llaves_a & {f["gemelo_de"] or f["id"] for f in b})


if __name__ == "__main__":
    unittest.main()
