package allumettes;

public interface Strategie {
	/** Obtenir le nombre d'allumettes à retirer par la stratégie à partir du jeu.
	 * @param jeu jeu auquel on va appliquer le retrait d'allumettes
	 * @return nombre d'allumettes retirées
	 */
	int priseStrategie(Jeu jeu);
}
