package allumettes;

public class StrategieRapide implements Strategie {

	/** Construire la stratégie.
	 */
	public StrategieRapide() {
	}


	@Override
	public int priseStrategie(Jeu jeu) {
		final int priseMax = 3;
		int prise;

		// Cas où le nombre d'allumettes est strictement inférieur au nombre maximum
		// de retrait autorisé par les règles du jeu
		if (jeu.getNombreAllumettes() < priseMax) {
			prise = jeu.getNombreAllumettes();
		} else { 	// Sinon le joueur rapide prend le nombre maximum d'allumettes
					// autorisées par les règles du jeu
			prise = priseMax;
		}
		return prise;
	}

}
