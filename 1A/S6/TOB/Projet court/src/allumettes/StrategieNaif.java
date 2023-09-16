package allumettes;

import java.util.Random;

public class StrategieNaif implements Strategie {

	/** Construire la stratégie.
	 */
	public StrategieNaif() {
	}

	@Override
	public int priseStrategie(Jeu jeu) {
		final int priseMax = Jeu.PRISE_MAX;
		Random random = new Random();
		int prise;

		// On traite les cas où le nombre d'allumettes restant est strictement
		// inférieur au nombre maximum de retrait autorisé par les règles du jeu
		if (jeu.getNombreAllumettes() == priseMax - 1) {
			prise = random.nextInt(priseMax - 1) + 1;
		} else if (jeu.getNombreAllumettes() == 1) {
			prise = 1;
		} else {
			prise = random.nextInt(priseMax) + 1;
		}
		return prise;
	}
}

