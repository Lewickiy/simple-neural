import com.lewickiy.util.Activation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.assertTrue;

@RunWith(Parameterized.class)
public class ActivationTest {

    @Parameterized.Parameter
    public double value;

    @Parameterized.Parameters(name = "value {0} ranges from 0.0 to 1.0")
    public static Object[] testData() {
        return new Object[] {13.0, 17.6, 0.26, 123.76, 100.0, 18700.926
                , 0.00002, -12,75, -817.2443, -7342,765, 123746.297856
                , 1789232.0000011
        };
    }

    @Test
    public void derivativeSigmoidMustReturnDoubleFromZeroToOne() {
        double actual = Activation.derivativeSigmoid(value);
        boolean result = actual <= 1.0 && actual >= 0.0;
        assertTrue(result);
    }
}
