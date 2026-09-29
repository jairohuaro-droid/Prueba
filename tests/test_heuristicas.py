import unittest

from detector.heuristicas import analizar

TEXTO_IA = (
    "En el panorama actual, la tecnología desempeña un papel crucial en la educación. "
    "Es importante destacar que no solo mejora el acceso, sino también la calidad. "
    "Cabe señalar que los docentes deben adaptarse a estos cambios. "
    "En conclusión, es fundamental adoptar un enfoque equilibrado."
)

TEXTO_HUMANO = (
    "Ayer se me rompió la bici. Otra vez. "
    "Iba bajando por la calle de mi tía, la que tiene el perro que ladra a todo lo que se mueve, "
    "y de repente la cadena hizo un ruido rarísimo y se salió. "
    "Tuve que volver caminando."
)


class TestHeuristicas(unittest.TestCase):
    def test_texto_ia_puntua_mas_alto(self):
        self.assertGreater(analizar(TEXTO_IA)["puntuacion_ia"], analizar(TEXTO_HUMANO)["puntuacion_ia"])

    def test_detecta_marcadores(self):
        self.assertIn("en conclusión", analizar(TEXTO_IA)["marcadores_encontrados"])

    def test_detecta_invisibles_y_homoglifos(self):
        s = analizar("hola​ mundo cаsa")["senales"]  # 'а' cirílica
        self.assertEqual(s["caracteres_invisibles"], 1)
        self.assertEqual(s["homoglifos"], 1)

    def test_texto_corto_no_confiable(self):
        self.assertFalse(analizar(TEXTO_HUMANO)["confiable"])


if __name__ == "__main__":
    unittest.main()
