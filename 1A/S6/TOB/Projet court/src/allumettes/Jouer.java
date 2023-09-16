package allumettes;

import java.util.Scanner;

/**
 * Lance une partie des 13 allumettes en fonction des arguments fournis sur la
 * ligne de commande.
 *
 * @author Xavier Crégut
 * @version $Revision: 1.5 $
 */
public class Jouer {

	/**
	 * Lancer une partie. En argument sont donnés les deux joueurs sous la forme
	 * nom@stratégie.
	 *
	 * @param args la description des deux joueurs
	 */
	public static void main(String[] args) throws CoupInvalideException {
		try {
			final int nbJoueurs = 2;

			// On vérifie qu'il y a le même nombre d'arguments que de joueurs
			// ou que le même nombre d'arguments que de joueurs + 1
			verifierNombreArguments(args);

			// On déclare les variables qui permettront de définir les caractéristiques
			// des joueur (le nom et la stratégie)
			String nomJoueur1;
			String nomJoueur2;
			String nomStrategie1;
			String nomStrategie2;
			Strategie strategie1;
			Strategie strategie2;

			// Déclaration de la partie réelle
			Partie partieReelle;

			// "Ouverture" d'un scanner (utile pour la stratégie humain)
			Scanner scanner = new Scanner(System.in);

			// Déclaration d'un booléen permettant de savoir si l'arbitre est confiant
			boolean arbitreConfiant = false;
			if (args.length == nbJoueurs + 1) {

				// Attribution du type d'arbitre, ici confiant (si le premier argument,
				// la partie sera lancée comme si l'utilisateur n'avait rien mis,
				// l'arbitre ne sera donc pas confiant)
				arbitreConfiant = args[0].equals("-confiant");


				String[] identite1 = args[1].split("@");
				String[] identite2 = args[2].split("@");

				// On vérifie qu'il y a bien nom1@strategie1 pour le premier joueur
				verifierFormatIdentite(identite1);

				nomJoueur1 = identite1[0];
				nomStrategie1 = identite1[1];

				// On vérifie qu'il y a bien nom2@strategie2 pour le deuxième joueur
				verifierFormatIdentite(identite2);

				nomJoueur2 = identite2[0];
				nomStrategie2 = identite2[1];
			} else {

				// Attribution du type d'arbitre, ici pas confiant
				arbitreConfiant = false;

				// Décomposition de joueur@strategie (les variables identite sont des
				// tableaux de String contenant le nom puis la stratégie)
				String[] identite1 = args[0].split("@");
				String[] identite2 = args[1].split("@");

				// On vérifie qu'il y a bien nom1@strategie1 pour le premier joueur
				verifierFormatIdentite(identite1);

				nomJoueur1 = identite1[0];
				nomStrategie1 = identite1[1];

				// On vérifie qu'il y a bien nom2@strategie2 pour le deuxième joueur
				verifierFormatIdentite(identite2);

				nomJoueur2 = identite2[0];
				nomStrategie2 = identite2[1];
			}

			partieReelle = new Partie();

			// Attribution des stratégies de chaque joueur à partir de celles rentrées
			// en arguments de l'exécution
			strategie1 = attribuerStrategie(nomJoueur1, nomStrategie1, scanner);
						// scanner est utilisé uniquement dans le cas d'un humain
			strategie2 = attribuerStrategie(nomJoueur2, nomStrategie2, scanner);
						// scanner est utilisé uniquement dans le cas d'un humain

			// Construction des deux joueurs
			Joueur joueur1 = new Joueur(nomJoueur1, strategie1);
			Joueur joueur2 = new Joueur(nomJoueur2, strategie2);

			// Construction de l'arbitre
			Arbitre arbitre = new Arbitre(joueur1, joueur2);

			// Paramétrage de la confiance de l'arbitre
			arbitre.setArbitreConfiant(arbitreConfiant);

			// Arbitrage de partieReelle
			arbitre.arbitrer(partieReelle);

			scanner.close();	// On pense à fermer le scanner
			if (arbitre.getInterruption()) {
				return;
			} else {
				System.out.println(arbitre.getPerdant().getNom() + " perd !");
				System.out.println(arbitre.getGagnant().getNom() + " gagne !");
			}

		} catch (ConfigurationException e) {
			System.out.println();
			System.out.println("Erreur : " + e.getMessage());
			afficherUsage();
			System.exit(1);
		} catch (ArrayIndexOutOfBoundsException aiobe) {
			System.out.println();
			System.out.println("Erreur : " + aiobe.getMessage());
			afficherUsage();
		}
	}

	/** Renvoie la stratégie d'un joueur à partir du nom de la stratégie
	 * et du nom du joueur
	 * @param nomJoueur nom du joueur (important pour la stratégie humain)
	 * @param nomStrategie nom de la stratégie
	 */
	public static Strategie attribuerStrategie(String nomJoueur, String nomStrategie,
			Scanner scanner) {
		Strategie strategie;
		switch (nomStrategie) {
		case "rapide":
			strategie = new StrategieRapide();
			break;
		case "naif":
			strategie = new StrategieNaif();
			break;
		case "expert":
			strategie = new StrategieExpert();
			break;
		case "humain":
			strategie = new StrategieHumain(nomJoueur, scanner);
			break;
		case "tricheur":
			strategie = new StrategieTricheur();
			break;
		default:
			throw new ConfigurationException("La stratégie N'EXISTE PAS");
		}
		return strategie;
	}

	/** Vérifier le nombre d'aguments avec lequel on exécute le programme
	 * @param args la description des deux joueurs
	 */
	private static void verifierNombreArguments(String[] args) {
		final int nbJoueurs = 2;
		if (args.length < nbJoueurs) {
			throw new ConfigurationException("Trop peu d'arguments : " + args.length);
		}
		if (args.length > nbJoueurs + 1) {
			throw new ConfigurationException("Trop d'arguments : " + args.length);
		}
	}

	/** Vérifier que le format donnée pour un joueur est bonne (nom@strategie)
	 * @param args la description d'un joueur
	 */
	private static void verifierFormatIdentite(String[] args) {
		final int nbArgId = 2;
		if (args.length < nbArgId) {
			throw new ConfigurationException("Trop peu d'arguments "
					+ "pour l'un des joueurs : " + args.length);
		}
		if (args.length > nbArgId) {
			throw new ConfigurationException("Trop d'arguments "
					+ "pour l'un des joueurs : " + args.length);
		}
	}

	/** Afficher des indications sur la manière d'exécuter cette classe. */
	public static void afficherUsage() {
		System.out.println("\n" + "Usage :" + "\n\t"
				+ "java allumettes.Jouer joueur1 joueur2" + "\n\t\t"
				+ "joueur est de la forme nom@stratégie" + "\n\t\t"
				+ "strategie = naif | rapide | expert | humain | tricheur"
				+ "\n" + "\n\t" + "Exemple :" + "\n\t"
				+ "	java allumettes.Jouer Xavier@humain " + "Ordinateur@naif"
				+ "\n");
	}

}
