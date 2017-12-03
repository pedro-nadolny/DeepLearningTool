package deeplearning.controllers;

import com.jfoenix.controls.JFXSpinner;
import deeplearning.models.HistoryRow;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;

import java.text.DateFormat;
import java.text.SimpleDateFormat;

public class HistoryListViewCell extends ListCell<HistoryRow> {
    @FXML private Label title;
    @FXML private ImageView iconFail;
    @FXML private ImageView iconDone;
    @FXML private JFXSpinner spinner;
    @FXML private HBox box;

    private  FXMLLoader loader;

    @Override
    protected void updateItem(HistoryRow item, boolean empty) {
        super.updateItem(item, empty);

        if(empty || item == null) {
            setText(null);
            setGraphic(null);
            return;
        }

        if(loader == null) {
            loader = new FXMLLoader(getClass().getClassLoader().getResource("HistoryListViewCell.fxml"));
            loader.setController(this);

            try {
                loader.load();
            } catch (Exception e) {
                e.printStackTrace();
            }

            spinner.setRadius(10);
        }

        box.getChildren().remove(spinner);
        box.getChildren().remove(iconFail);
        box.getChildren().remove(iconDone);

        StringBuilder titleStr = new StringBuilder();

        if(item.enabledDL) {
            titleStr.append("DL ");

            if(item.enabledCC) {
                titleStr.append("+ ");
            }
        }

        if(item.enabledCC) {
            switch (item.classicalType) {
                case NB: titleStr.append("NB ");
                    break;

                case NN: titleStr.append("NN ");
                    break;

                case RB: titleStr.append("RB ");
                    break;

                case C45: titleStr.append("C45 ");
                    break;
            }
        }

        DateFormat dateFormat = new SimpleDateFormat("@ HH:mm:ss");
        titleStr.append(dateFormat.format(item.startTime));
        title.setText(titleStr.toString());

        boolean ended = item.endedCC.get() && !item.enabledDL;
        ended = ended || (item.endedDL.get() && !item.enabledCC);
        ended = ended || (item.endedDL.get() && item.endedCC.get());

        boolean fail = item.failDL.get() && (item.endedCC.get() || item.failCC.get() || !item.enabledCC) ||
                       item.failCC.get() && (item.endedDL.get() || item.failDL.get() || !item.enabledDL);

        if (ended) {
            box.getChildren().add(0, iconDone);
        } else if(fail) {
            box.getChildren().add(0, iconFail);
        } else {
            box.getChildren().add(0, spinner);
        }

        setText(null);
        setGraphic(box);
    }
}
