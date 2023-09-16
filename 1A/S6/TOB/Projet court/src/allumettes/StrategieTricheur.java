package allumettes;

public class StrategieTricheur implements Strategie {

	/** Construire la stratégie.
	 */
	public StrategieTricheur() {
	}

	@Override
	public int priseStrategie(Jeu jeu) throws OperationInterditeException {
		System.out.println("[Je triche...]");

		// On retire des allumettes jusqu'à ce qu'il n'en reste plus que 2
		while (jeu.getNombreAllumettes() > 2) {
			try {
				jeu.retirer(1);
			} catch (CoupInvalideException ie) {
			} // n'arrive jamais a priori
		}
		if (jeu.getNombreAllumettes() == 2) {
			System.out.println("[Allumettes restantes : 2]");
		} else { 	// Cas a priori inutile (sauf peut-être si un joueur prend
				// stratégie tricheur quand il ne reste qu'une allumette en jeu)
			System.out.println("[Impossible de tricher !]");
		}
		return 1;
	}
}
