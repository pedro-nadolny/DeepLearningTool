package deeplearning.models;

import javafx.beans.InvalidationListener;
import javafx.beans.Observable;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.util.Callback;
import weka.classifiers.Evaluation;

/**
 * Created by Pedro on 6/28/17.
 */
public class HistoryRow {
    public StringProperty title = new SimpleStringProperty();
    public BooleanProperty fail = new SimpleBooleanProperty();
    public BooleanProperty isRunningDL = new SimpleBooleanProperty();
    public BooleanProperty isRunningCC = new SimpleBooleanProperty();
    public BooleanProperty endedCC = new SimpleBooleanProperty();
    public BooleanProperty endedDL = new SimpleBooleanProperty();

    public ObjectProperty<Evaluation> evalDL = new SimpleObjectProperty<Evaluation>();
    public ObjectProperty<Evaluation> evalCC = new SimpleObjectProperty<Evaluation>();

    public HistoryRow(String title, boolean isRunningDL, boolean isRunningCC) {
        this.title.set(title);
        this.fail.set(false);
        this.isRunningDL.set(isRunningDL);
        this.isRunningCC.set(isRunningCC);
        this.endedCC.set(false);
        this.endedDL.set(false);
    }

    public static Callback<HistoryRow, Observable[]> extractor() {
        return new Callback<HistoryRow, Observable[]>() {
            @Override
            public Observable[] call(HistoryRow param) {
                return new Observable[]{param.title, param.fail, param.endedCC, param.endedDL, param.isRunningDL, param.isRunningCC};
            }
        };
    }
}
