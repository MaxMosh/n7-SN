package allumettes;

public class Arbitre {

	private Joueur joueur1;
	private Joueur joueur2;
	private Joueur gagnant;
	private Joueur perdant;
	private boolean arbitreConfiant; 	// booléen qui nous permet de savoir si
										// l'arbitre est confiant ou non
	private boolean interruption; 		// booléen qui nous permet de savoir en aval
										// s'il y a eu un problème et que la partie
										// a du être interrompue

	public Arbitre(Joueur j1, Joueur j2) {
		this.joueur1 = j1;
		this.joueur2 = j2;
		// this.arbitreConfiant = arbitreConfiant;
		this.interruption = false; 	// cet attribut reste à false s'il n'y pas de triche
								// détectée et que la partie n'est donc pas interrompue
	}

	public void setArbitreConfiant(boolean arbitreConfiant) {
		this.arbitreConfiant = arbitreConfiant;
	}

	public void arbitrer(Jeu jeu) throws CoupInvalideException {
		// On initialise le joueur courant au joueur 1, ainsi que le booléen nous
		// permettant d'indiquer à quel joueur de jouer
		Joueur joueurCourant = this.joueur1;
		boolean aJ1DeJouer = true;

		int allumettesARetirer;
		while (jeu.getNombreAllumettes() > 0) {

			// Afficher les allumettes restantes
			System.out.println("Allumettes restantes : " + jeu.getNombreAllumettes());

			// Proposer au joueur combien il prend d'allumettes
			// dans le cas où il est humain
			if (joueurCourant.getStrategie() instanceof StrategieHumain) {
										// 4 est l'identifiant de la stratégie humain
				System.out.print(joueurCourant.getNom() + ", combien d'allumettes ? ");
			}

			// Le joueur entre le nombre d'allumettes qu'il veut prendre,
			// on lui donne accès au jeu réel ou à la procuration selon
			// si l'arbitre est confiant ou non.
			if (this.arbitreConfiant) {
				allumettesARetirer = joueurCourant.getPrise(jeu);
			} else {
				PartieProxy partieProcuration = new PartieProxy(jeu);
				try {
					allumettesARetirer = joueurCourant.getPrise(partieProcuration);
				} catch (OperationInterditeException oie) { // Exception dans le cas où
										// l'arbitre pas confiant détecte une triche
					System.out.println("Abandon de la partie car "
										+ joueurCourant.getNom() + " triche !");
					this.interruption = true;
					return;
				}
			}

			try {
				// On retire le nombre d'allumettes que le joueur souhaite retirer
				jeu.retirer(allumettesARetirer);
				switch (allumettesARetirer) {
				case 1:
					System.out.println(joueurCourant.getNom()
										+ " prend 1 allumette. \n");
					break;
				default:
					System.out.println(joueurCourant.getNom() + " prend "
										+ allumettesARetirer + " allumettes. \n");
				}

				// On change de joueur dans le cas où il reste des coups à jouer
				if (jeu.getNombreAllumettes() > 0) {
					aJ1DeJouer = !aJ1DeJouer;
					if (aJ1DeJouer) {
						joueurCourant = this.joueur1;
					} else {
						joueurCourant = this.joueur2;
					}
				}

			} catch (CoupInvalideException ie) { // Exception dans le cas où le
									//nombre d'allumettes à retirer n'est pas bon
				switch (allumettesARetirer) {
				case 0:
				case -1:
					System.out.println(joueurCourant.getNom() + " prend "
											+ allumettesARetirer + " allumette.");
					break;
				default:
					System.out.println(joueurCourant.getNom() + " prend "
											+ allumettesARetirer + " allumettes.");
				}
				System.out.println("Impossible ! Nombre invalide : "
									+ ie.getCoup() + " (" + ie.getProbleme() + ") \n");

				// Pour effectuer un appel récursif de la méthode arbitrer, on s'assure
				// de faire
				// rejouer le joueur courant
				if (joueurCourant.getNom() == joueur2.getNom()) {
										// Dans le cas où joueur2 est joueur courant, on
										// intervertit donc joueur1 et joueur2
					this.joueur2 = this.joueur1;
					this.joueur1 = joueurCourant;
					aJ1DeJouer = !aJ1DeJouer; // On n'oublie pas d'inverser le booléen !
				}
			}
		}

		// On met à jour le gagnant et le perdant en fin de partie (le perdant
		// est le joueur courant, l'autre joueur est gagnant)
		if (aJ1DeJouer) {
			this.gagnant = this.joueur2;
			this.perdant = this.joueur1;
		} else {
			this.gagnant = this.joueur1;
			this.perdant = this.joueur2;
		}
	}

	/** Obtenir le gagnant retenu par l'arbitre.
	 * @return joueur gagnant
	 */
	public Joueur getGagnant() {
		return this.gagnant;
	}

	/** Obtenir le perdant retenu par l'arbitre.
	 * @return joueur perdant
	 */
	public Joueur getPerdant() {
		return this.perdant;
	}

	/** Savoir si la partie a du être interrompue à cause de triche.
	 * @return booléen permettant de savoir si la partie a été interrompu (true s'il
	 * y a eu interruption)
	 */
	public boolean getInterruption() {
		return this.interruption;
	}
}
