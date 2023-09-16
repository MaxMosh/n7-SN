import java.awt.Color;
import org.junit.*;
import static org.junit.Assert.*;

/**
  * Classe de test complémentaire de la classe Cercle (exigences E12, E13 et E14).
  * @author	Maxime Moshfeghi
  * @version	$??$
  */
public class CercleTest {

	// précision pour les comparaisons réelle
	public final static double EPSILON = 0.001;

	// Les points du sujet
	private Point A1, A2, A3;

	// Les cercles du sujet
	private Cercle C1, C2, C3;

	@Before public void setUp() {
		// Construire les points
		A1 = new Point(1, 1);
		A2 = new Point(3, 3);
		A3 = new Point(2, 2);

		// Construire les cercles
		C1 = new Cercle(A1, A2);
		C2 = new Cercle(A1, new Point(6, 1), Color.orange);
		C3 = Cercle.creerCercle(A1, A3);
	}

	/** Vérifier si deux points ont mêmes coordonnées.
	  * @param p1 le premier point
	  * @param p2 le deuxième point
	  */
	static void memesCoordonnees(String message, Point p1, Point p2) {
		assertEquals(message + " (x)", p1.getX(), p2.getX(), EPSILON);
		assertEquals(message + " (y)", p1.getY(), p2.getY(), EPSILON);
	}

	@Test public void testerE12() {
		memesCoordonnees("Le centre de C1 est bien initialisé", C1.getCentre(), new Point(2, 2));
		assertEquals("E12 : Rayon de C1 correct",
				Math.sqrt(2), C1.getRayon(), EPSILON);
		assertEquals(Color.blue, C1.getCouleur());
	}

	@Test public void testerE13() {
		memesCoordonnees("Le centre de C2 n'est pas bien initialisé", C2.getCentre(), new Point(3.5, 1));
		assertEquals("E13 : Rayon de C2 correct",
				2.5, C2.getRayon(), EPSILON);
		assertEquals(Color.orange, C2.getCouleur());
	}

	@Test public void testerE14() {
		memesCoordonnees("Le centre de C3 n'est pas bien initialisé", C3.getCentre(), A1);
		assertEquals("E14 : Rayon de C3 correct",
				A1.distance(A3), C3.getRayon(), EPSILON); 
	}
}

//MODIF POUR PUSH