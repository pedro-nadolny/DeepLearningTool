package deeplearning.models;

import javafx.beans.InvalidationListener;
import javafx.beans.Observable;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.util.Callback;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Date;

/**
 * Created by Pedro on 6/28/17.
 */
public class HistoryRow {
    public BooleanProperty fail = new SimpleBooleanProperty(false);
    public BooleanProperty endedCC = new SimpleBooleanProperty(false);
    public BooleanProperty endedDL = new SimpleBooleanProperty(false);

    public Evaluation evalDL;
    public Evaluation evalCC;

    public boolean isRunningDL;
    public boolean isRunningCC;

    public Classifier classifier;

    public CCType classicalType;

    public Date startTime;

    public HistoryRow(boolean isRunningDL, boolean isRunningCC) {
        this.isRunningDL = isRunningDL;
        this.isRunningCC = isRunningCC;
        this.startTime = new Date();
    }

    public static Callback<HistoryRow, Observable[]> extractor() {
        return new Callback<HistoryRow, Observable[]>()  {
            @Override
            public Observable[] call(HistoryRow param) {
                return new Observable[]{param.endedCC, param.endedDL, param.fail};
            }
        };
    }

    public enum CCType {
        C45(0), NN(1), RB(2), NB(3);

        public int type;
        CCType(int t) {
            this.type = t;
        }
    }
}
