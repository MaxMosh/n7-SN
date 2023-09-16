package allumettes;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

public class TestDebut {
	private Strategie strategie;

	@Before
	public void setUp() {
		this.strategie = new StrategieExpert();
	}

	@Test
	public void testStrategieExpert() {
		assertEquals(this.strategie.priseStrategie(new Partie(4)), 3);
		assertEquals(this.strategie.priseStrategie(new Partie(3)), 2);
		assertEquals(this.strategie.priseStrategie(new Partie(2)), 1);
		assertEquals(this.strategie.priseStrategie(new Partie(8)), 3);
	}
}
