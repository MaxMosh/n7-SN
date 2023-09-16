package allumettes;

import java.util.Random;

public class StrategieExpert implements Strategie {

	/** Construire la stratégie.
	 */
	public StrategieExpert() {
	}

	@Override
	public int priseStrategie(Jeu jeu) {

		// On veut laisser à l'adversaire 1 mod 4 allumettes
		int prise;

		// S'il reste une allumette l'expert est coincé
		if (jeu.getNombreAllumettes() == 1) {
			prise = 1;
		} else if (jeu.getNombreAllumettes() % (Jeu.PRISE_MAX + 1) == 1) {
				// Si il reste 1 mod 4 allumettes (valeur autre que 1, il n'existe
				// a priori pas de coup avantageux pour l'expert, il prend un nombre
				// alétoire d'allumettes
			Random random = new Random();
			prise = random.nextInt(Jeu.PRISE_MAX) + 1;
		} else if (jeu.getNombreAllumettes() % (Jeu.PRISE_MAX + 1) == 0) {
				// Autrement, l'expert peut laisser 1 mod 4 allumettes à l'adversaire
				// et il va gagner
			prise = Jeu.PRISE_MAX;
		} else {
			prise = jeu.getNombreAllumettes() % (Jeu.PRISE_MAX + 1) - 1;
		}
		return prise;
	}
}
