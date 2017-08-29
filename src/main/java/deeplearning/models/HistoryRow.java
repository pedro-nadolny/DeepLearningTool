package deeplearning.models;

import javafx.beans.Observable;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.util.Callback;
import weka.classifiers.Evaluation;

/**
 * Created by Pedro on 6/28/17.
 */
public class HistoryRow {
    public StringProperty title = new SimpleStringProperty();
    public BooleanProperty done = new SimpleBooleanProperty();
    private Evaluation eval;

    public HistoryRow(String title, Boolean done) {
        this.title = new SimpleStringProperty(title);
        this.done = new SimpleBooleanProperty(done);
    }

    public static Callback<HistoryRow, Observable[]> extractor() {
        return new Callback<HistoryRow, Observable[]>() {
            @Override
            public Observable[] call(HistoryRow param) {
                return new Observable[]{param.title, param.done};
            }
        };
    }

    public void setEvaluation(Evaluation e) {
        this.eval = e;
    }
}
