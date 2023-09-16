package allumettes;

public class PartieProxy implements Jeu {

	public static final int NB_INIT = 13; // Constante commune
	private Jeu jeu;

	/** Construire une procuration à partir du jeu réel.
	 * @param jeu jeu réel
	 */
	public PartieProxy(Jeu jeu) {
		this.jeu = jeu;
	}

	@Override
	public void retirer(int nbPrises) throws CoupInvalideException {
		throw new OperationInterditeException("TRICHE !");
	}

	@Override
	public int getNombreAllumettes() {
		return this.jeu.getNombreAllumettes();
	}

}
