package allumettes;

public class Partie implements Jeu {

	public static final int NB_INIT = 13; // Constante commune
	private int nbRestant;

	/** Construire une partie. Le nombre initial d'allumettes est défini à NB_INIT.
	 */
	public Partie() {
		this.nbRestant = NB_INIT;
	}

	/** Construire une partie à partir d'un nombre initial d'allumettes choisi.
	 * @param nbInitUsager nombre d'allumettes initial
	 */
	public Partie(int nbInitUsager) {
		this.nbRestant = nbInitUsager;
	}

	@Override
	public int getNombreAllumettes() {
		return this.nbRestant;
	}

	@Override
	public void retirer(int nbPrises) throws CoupInvalideException {
		if (nbPrises > this.nbRestant) {
			throw new CoupInvalideException(nbPrises, "> " + this.getNombreAllumettes());
		} else if (nbPrises < 1) {
			throw new CoupInvalideException(nbPrises, "< 1");
		} else if (nbPrises > PRISE_MAX) {
			throw new CoupInvalideException(nbPrises, "> " + Jeu.PRISE_MAX);
		}
		this.nbRestant -= nbPrises;
	}

}
