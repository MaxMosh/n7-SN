package allumettes;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

public class StrategieRapideTest {
	private Strategie strategie;

	@Before
	public void setUp() {
		this.strategie = new StrategieRapide();
	}

	@Test
	public void testStrategieRapide() {
		assertEquals(this.strategie.priseStrategie(new Partie()), 3);
		assertEquals(this.strategie.priseStrategie(new Partie(10)), 3);
		assertEquals(this.strategie.priseStrategie(new Partie(4)), 3);
		assertEquals(this.strategie.priseStrategie(new Partie(3)), 3);
		assertEquals(this.strategie.priseStrategie(new Partie(2)), 2);
		assertEquals(this.strategie.priseStrategie(new Partie(1)), 1);
		assertEquals(this.strategie.priseStrategie(new Partie()), 3);
	}
}
