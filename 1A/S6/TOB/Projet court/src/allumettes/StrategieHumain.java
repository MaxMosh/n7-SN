package allumettes;

import java.util.Scanner;

public class StrategieHumain implements Strategie {
	private String nomJoueur;
	private Scanner scanner;

	/** Construire la stratégie à partir du nom du joueur et d'un scanner.
	 * @param nomJoueur nom du joueur que l'on souhaite construire
	 * @param scanner scanner que l'on donne à la stratégie pour pouvoir entrer le
	 * nombre d'allumettes à retirer
	 */
	public StrategieHumain(String nomJoueur, Scanner scanner) {
		this.nomJoueur = nomJoueur;
		this.scanner = scanner;
	}

	/** Obtenir le nom du joueur (important pour l'affichage de messages).
	 * @return nom du joueur
	 */
	public String getNomJoueur() {
		return this.nomJoueur;
	}

	@Override
	public int priseStrategie(Jeu jeu) throws OperationInterditeException {

		String priseStr = this.scanner.next();
		try {

			// Cas où l'humain triche
			if (priseStr.equals("triche")) {
				try {
					// On traite le cas où il ne reste qu'une seule allumettes, de sorte
					// à sortir immédiatement de la fonction dans ce cas.
					if (jeu.getNombreAllumettes() == 1) {
						return 1;
					} else { // Sinon, on retire "discrètrement" une allumette
						jeu.retirer(1);
						System.out.println(
								"[Une allumette en moins, plus que "
								+ jeu.getNombreAllumettes() + ". Chut !]");
						System.out.print(this.getNomJoueur()
								+ ", combien d'allumettes ?");
						return this.priseStrategie(jeu);
					}
				} catch (CoupInvalideException ie) { 	// Le catch est a priori inutile,
														// on ne rentre jamais dans ce
														// cas.
					System.out.println("Plus assez d'allumettes pour tricher !");
					return this.priseStrategie(jeu);
				}
			} else { // Cas où l'humain souhaite retirer une valeur entière d'allumettes
				int prise = Integer.parseInt(priseStr);
				return prise;
			}
		} catch (NumberFormatException nfe) { 	// Cas où la chaine de caractères n'est
												//	pas "-triche" ou ne contient pas que
												// des chiffres
			System.out.println("Vous devez donner un entier.");
			System.out.print(this.getNomJoueur() + ", combien d'allumettes ? ");
			return this.priseStrategie(jeu);
		}

	}
}
