import java.awt.Color;

/** Cercle modélise un cercle dans un plan équipé d'un
 * repère cartésien. Un cercle peut être affiché et translaté.
 * On peut obtenir un certains nombres d'informatiops.
 *
 * @author  Maxime Moshfeghi
 */
public class Cercle implements Mesurable2D {
	public final static double PI = Math.PI;

	private Point centre;		//Centre du cercle
	private double rayon;		//Rayon du cercle
	private Color couleur;		//Couleur du cercle

	/** Construire un cercle à partir de son centre et de son rayon. La couleur est initialisée
	 * par défaut à bleue.
	 * @param O centre
	 * @param r rayon
	 */
	public Cercle(Point o, double r) {
		// préconditions
		assert r > 0;
		assert o != null;
		//
		this.centre = new Point(o.getX(),o.getY());
		this.rayon = r;
		this.couleur = Color.blue;
	}

	/** Construire un cercle à partir de deux points diamétralement opposés du cercle. La couleur
	 * est initialisée par défaut à bleue.
	 * @param a1 Premier point extrémal
	 * @param a2 Deuxième point extrémal
	 */
	public Cercle(Point a1, Point a2) {
		// préconditions
		assert a1 != null;
		assert a2 != null;
		assert (a1.getX() != a2.getX()) || (a1.getY() != a2.getY());
		//
		this.centre = new Point((a1.getX() + a2.getX()) / 2, (a1.getY() + a2.getY()) / 2);
		Point a1Copie = new Point(a1.getX(), a1.getY());
		Point a2Copie = new Point(a2.getX(), a2.getY());
		this.rayon = a1Copie.distance(a2Copie) / 2;
		this.couleur = Color.blue;
	}

	/** Construire un cercle à partir de deux points diamétralement opposés du cercle, ainsi
	 * que de sa couleur.
	 * @param a1 Premier point extrémal
	 * @param a2 Deuxième point extrémal
	 * @param couleur Couleur du cercle
	 */
	public Cercle(Point a1, Point a2, Color couleur) {
		// préconditions
		assert a1 != null;
		assert a2 != null;
		assert (a1.getX() != a2.getX()) || (a1.getY() != a2.getY());
		assert couleur != null;
		//
		this.centre = new Point((a1.getX() + a2.getX()) / 2, (a1.getY() + a2.getY()) / 2);
		Point a1Copie = new Point(a1.getX(), a1.getY());
		Point a2Copie = new Point(a2.getX(), a2.getY());
		this.rayon = a1Copie.distance(a2Copie) / 2;
		this.couleur = couleur;
	}

	/** Translater le cercle.
	 * @param tx déplacement suivant l'axe des X
	 * @param ty déplacement suivant l'axe des Y
	 */
	public void translater(double tx, double ty) {
		this.centre.translater(tx, ty);
	}

	/** Obtenir le centre du cercle.
	 * @return centre du cercle
	 */
	public Point getCentre() {
		return new Point(this.centre.getX(), this.centre.getY());
	}

	/** Obtenir le rayon du cercle.
	 * @return rayon du cercle
	 */
	public double getRayon() {
		return this.rayon;
	}

	/** Obtenir le diamètre du cercle.
	 * @return diamètre du cercle
	 */
	public double getDiametre() {
		return 2 * this.rayon;
	}

	/** Savoir si un point est à l'intérieur du cercle (au sens large).
	 * @param point A à tester
	 * @return booléen précisant si le point A est dans le cercle
	 */
	public boolean contient(Point a) {
		// préconditions
		assert a != null;
		//
		return this.centre.distance(a) <= this.rayon;
	}

	/** Obtenir le permètre du cercle.
	 * @return perimètre du cercle
	 */
	public double perimetre() {
		return 2 * PI * this.rayon;
	}

	/** Obtenir l'aire du cercle.
	 * @return aire du cercle
	 */
	public double aire() {
		return PI * this.rayon * this.rayon;
	}

	/** Obtenir la couleur du cercle.
	 * @return couleur du cercle
	 */
	public Color getCouleur() {
		return this.couleur;
	}

	/** Changer la couleur du cercle.
	 * @param nouvelle couleur du cercle
	 */
	public void setCouleur(Color couleur) {
		assert couleur != null;
		this.couleur = couleur;
	}

	/** Méthode pour créer un cercle à partur de son centre et d'un point qui lui appartient.
	 * @param centre du cercle
	 * @param point sur le cercle
	 * @return cercle créé
	 */
	public static Cercle creerCercle(Point o, Point a) {
		// préconditions
		assert o != null;
		assert a != null;
		assert (o.getX() != a.getX()) || (o.getY() != a.getY());
		//
		Cercle c = new Cercle(o, o.distance(a));
		return c;
	}

	/** Afficher le cercle. */
	public String toString() {
		return "C" + this.rayon + "@" + this.centre;
	}

	// Afficher le cercle.
	public void afficher() {
		System.out.print(this);
	}

	/** Changer le rayon du cercle.
	  * @param r nouveau rayon du cercle
	  */
	public void setRayon(double r) {
		// préconditions
		assert r > 0;
		//
		this.rayon = r;
	}

	/** Changer le diamètre du cercle.
	  * @param d nouveau diamètre du cercle
	  */
	public void setDiametre(double d) {
		// préconditions
		assert d > 0;
		//
		this.rayon = d / 2;
	}


}

//MODIF POUR PUSH