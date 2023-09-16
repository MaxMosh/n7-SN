package allumettes;


public class Joueur {


	private String nom;
	private Strategie strategie;

	/** Construire un joueur à partir de son nom et de sa stratégie
	 * @param nomJoueur nom du joueur
	 * @param strategie stratégie du joueur que l'on souhaite construire
	 */
	public Joueur(String nom, Strategie strategie) {
		this.nom = nom;
		this.strategie = strategie;
	}


	/** Obtenir le nom du joueur.
	 * @return nom du joueur
	 */
	public String getNom() {
		return this.nom;
	}

	/** Obtenir la stratégie du joueur.
	 * @return stratégie du joueur
	 */
	public Strategie getStrategie() {
		return this.strategie;
	}

	/** Obtenir le nombre d'allumettes à retirer par le joueur.
	 * @param jeu jeu auquel on va appliquer le retrait d'allumettes
	 * @return stratégie du joueur
	 * @exception OperationInterditeException exception levée si le joueur tente de
	 * tricher avec un arbitre confiant
	 */
	public int getPrise(Jeu jeu) throws OperationInterditeException { // On donnera la
																// procuration au joueur
		return this.strategie.priseStrategie(jeu);
	}
}
