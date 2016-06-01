package il.ac.tau.adviplab.myimageproc;

import org.opencv.core.Point;

import java.util.Comparator;

/**
 * Created by iplab_4_2 on 6/1/2016.
 */
public class PointCompare
        implements Comparator<Point> {

    public int compare(final Point a, final Point b) {
        if (a.x < b.x) {
            return -1;
        }
        else if (a.x > b.x) {
            return 1;
        }
        else {
            return 0;
        }
    }
}