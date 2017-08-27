package deeplearning.controllers;

import com.jfoenix.controls.JFXSpinner;
import deeplearning.models.HistoryRow;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;

import java.io.IOException;
import java.rmi.server.ExportException;

public class HistoryListViewCell extends ListCell<HistoryRow> {
    @FXML private Label title;
    @FXML private ImageView icon;
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

        title.setText(item.title.get());

        box.getChildren().remove(spinner);
        box.getChildren().remove(icon);

        if (item.done.get()) {
            box.getChildren().add(0, icon);
        } else {

            box.getChildren().add(0, spinner);
        }

        setText(null);
        setGraphic(box);
    }
}
