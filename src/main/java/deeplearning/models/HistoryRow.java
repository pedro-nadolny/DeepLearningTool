package deeplearning.models;

import javafx.beans.InvalidationListener;
import javafx.beans.Observable;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.util.Callback;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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

    public org.deeplearning4j.eval.Evaluation evalDL;
    public Evaluation evalCC;

    public boolean isRunningDL;
    public boolean isRunningCC;

    public Classifier classifierCC;
    public MultiLayerNetwork classifierDL;

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

    public String getResultsTextCC() {
        StringBuilder str = new StringBuilder(this.classifierCC.toString());

        str.append("\n=== Classifier Statistics Summary ===\n" +evalCC.toSummaryString() + "\n");

        try {
            str.append(evalCC.toClassDetailsString());
            str.append("\n");
        } catch (Exception excp) {
            excp.printStackTrace();
        }

        try {
            str.append(evalCC.toMatrixString());
        } catch (Exception excp) {
            excp.printStackTrace();
        }

        return str.toString();
    }

    public String getResultsTextDL() {
        StringBuilder str = new StringBuilder();
        str.append("\n=== Classifier Statistics Summary ===\n" +evalDL.stats() + "\n" +classifierDL.summary());
        return str.toString();
    }

    public enum CCType {
        C45(0), NN(1), RB(2), NB(3);

        public int type;
        CCType(int t) {
            this.type = t;
        }
    }
}
