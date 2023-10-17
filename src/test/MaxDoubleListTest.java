import com.lewickiy.util.MaxDoubleList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class MaxDoubleListTest {
    @Parameterized.Parameter
    public double[] valuesList;
    @Parameterized.Parameter(1)
    public double expected;

    @Parameterized.Parameters(name = "Max value from {0} is - {1}")
    public static Object[][] getTestData() {
        return new Object[][] {
                {new double[] {0.89, 1.26, 0.73}, 1.26},
                {new double[] {0.64, 0.14, 0.02}, 0.64},
                {new double[] {0.24, 1.13, 1.99}, 1.99},
                {new double[] {0.0, 0.0, 0.01}, 0.01},
                {new double[] {-1.26, -7.234, 0.0}, 0.0}
        };
    }

    @Test
    public void shouldCheckPalindrome() {
        double actual = MaxDoubleList.max(valuesList);
        assertEquals(expected, actual, 0.001);
    }
}
