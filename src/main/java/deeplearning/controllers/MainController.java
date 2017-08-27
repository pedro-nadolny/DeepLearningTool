package deeplearning.controllers;

import com.jfoenix.controls.*;

import deeplearning.models.HistoryRow;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.rules.M5Rules;

import java.io.File;
import java.net.URL;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.ResourceBundle;

public class MainController extends VBox implements Initializable{

    @FXML private Label baseInfoAttributesLabel;
    @FXML private Label baseInfoNameLabel;
    @FXML private Label baseInfoClassesLabel;
    @FXML private Label baseInfoInstancesLabel;

    @FXML private JFXComboBox baseClassComboBox;
    @FXML private JFXListView<HistoryRow> historyListView;

    @FXML private JFXRadioButton radioButtonNeuralNetwork;
    @FXML private JFXRadioButton radioButtonRuleBased;
    @FXML private JFXRadioButton radioButtonNaiveBayes;
    @FXML private JFXRadioButton radioButtonC45;

    @FXML private JFXCheckBox checkBoxEnableDL;
    @FXML private JFXCheckBox checkBoxEnableCC;

    @FXML private StackPane root;

    private Instances base;
    private ObservableList<HistoryRow> historyList;

    private Thread mainThread = Thread.currentThread();

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        baseClassComboBox.setVisibleRowCount(6);

        historyList = FXCollections.observableArrayList(HistoryRow.extractor());
        historyListView.setItems(historyList);
        historyListView.setCellFactory(studentListView -> new HistoryListViewCell());
    }

    @FXML protected void handleLoadBase(ActionEvent event) throws Exception {

        File f = loadArff();

        if (f != null && f.getName().endsWith(".arff")) {

            DataSource source = new DataSource(f.getAbsolutePath());
            base = new Instances(source.getDataSet());

            Attribute classAttr = base.attribute("class");
            if(classAttr != null) {
                base.setClass(classAttr);
            } else {
                base.setClass(base.attribute(0));
            }

            updateBaseInfoSection();
        }
    }

    private File loadArff() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open ARFF file");
        FileChooser.ExtensionFilter arff = new FileChooser.ExtensionFilter("ARFF files (*.arf)",  "*.ARFF");
        fileChooser.getExtensionFilters().addAll(arff);

        return fileChooser.showOpenDialog(baseInfoClassesLabel.getScene().getWindow());
    }

    private void updateBaseInfoSection() {
        updateNameLabel();
        updateInstancesLabel();
        updateClassesLabel();
        updateAttributesLabel();
        updateClassesComboBox();
    }

    private void updateClassesComboBox() {
        ObservableList<String> options = FXCollections.observableArrayList();
        for(int i = 0; i < base.numAttributes(); i++) {
            options.add(base.attribute(i).name());
        }

        baseClassComboBox.setItems(options);
        baseClassComboBox.setDisable(false);
        baseClassComboBox.setValue(base.classAttribute().name());
    }

    private void updateNameLabel() {
        baseInfoNameLabel.setText("Name: " + base.relationName());
    }

    private void updateInstancesLabel() {
        StringBuilder builder = new StringBuilder("Instances: ");
        builder.append(base.numInstances());
        baseInfoInstancesLabel.setText(builder.toString());
    }

    private void updateAttributesLabel() {
        StringBuilder builder = new StringBuilder("Attributes: ");
        builder.append(base.numAttributes() - 1);
        baseInfoAttributesLabel.setText(builder.toString());
    }

    private void updateClassesLabel() {
        StringBuilder builder = new StringBuilder("Classes: ");

        if(!base.attribute(base.classIndex()).isNumeric()) {
            builder.append(base.numClasses());
        } else {
            builder.append("Not Discrete");
        }

        baseInfoClassesLabel.setText(builder.toString());
    }

    @FXML protected void handleClassComboBox(ActionEvent event) throws Exception {
        for(int i = 0; i < base.numAttributes(); i++) {
            if (base.attribute(i).name() == baseClassComboBox.getValue()) {
                base.setClassIndex(i);

                StringBuilder builder = new StringBuilder("Classes: ");

                if(base.classAttribute().isNominal()) {
                    builder.append(base.numClasses());
                } else {
                    builder.append("Not Discrete.");
                }
                baseInfoClassesLabel.setText(builder.toString());

                return;
            }
        }
    }

    @FXML protected void handleRunButton(ActionEvent event) throws Exception {

        boolean showDialog = false;
        JFXDialogLayout layout = new JFXDialogLayout();

        if(base == null) {
            showDialog = true;
            layout.setBody(new Text("Load a base file before you run." +
                    "\nYou can load a base file with the button on the top-left."));
        } else if (!checkBoxEnableDL.isSelected() && !checkBoxEnableCC.isSelected()) {
            showDialog = true;
            layout.setBody(new Text("Select a classification training method to use." +
                    "\nEnable at least one with the checkbox on either\nDEEP LEARNING or CLASSIC CLASSFICATION."));
        }

        if (showDialog) {
            JFXDialog dialog = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

            JFXButton closeButton = new JFXButton("Close");
            closeButton.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent event) {
                    dialog.close();
                }
            });

            layout.setActions(closeButton);
            dialog.show();

            return;
        }

        StringBuilder builder = new StringBuilder();

        if (checkBoxEnableCC.isSelected()) {
            if(checkBoxEnableDL.isSelected()) {
                builder.append("DL + ");
            }

            if (radioButtonC45.isSelected()) {
                builder.append("C4.5");
                addToHistory(builder.toString());
                runC45();

            } else if(radioButtonNaiveBayes.isSelected()) {
                builder.append("NB");
                addToHistory(builder.toString());
                runNaiveBayes();

            } else if (radioButtonNeuralNetwork.isSelected()) {
                builder.append("NN");
                addToHistory(builder.toString());
                runNeuralNetwork();

            } else if (radioButtonRuleBased.isSelected()) {
                builder.append("RB");
                addToHistory(builder.toString());
                runRuleBased();
            }
        }

        if (checkBoxEnableDL.isSelected()) {
            runDeepLearning();
        }
    }

    protected void addToHistory(String title) {
        StringBuilder builder = new StringBuilder(title);
        DateFormat dateFormat = new SimpleDateFormat(" @ HH:mm:ss");
        builder.append(dateFormat.format(new Date()));

        historyList.add(new HistoryRow(builder.toString(), false));
    }

    protected void runC45() throws Exception{
        Thread thread = new Thread(){
            public void run(){
                int historyIndex = historyList.size() - 1;
                J48 c45 = new J48();

                try {
                    c45.buildClassifier(base);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("Trace from C45 Thread.");
                }

                Platform.runLater(() -> finishTaskOfIndex(historyIndex));
            }
        };

        thread.start();
    }

    protected void runNaiveBayes() throws Exception {
        Thread thread = new Thread(){
            public void run(){
                int historyIndex = historyList.size() - 1;
                NaiveBayes NB = new NaiveBayes();

                try {
                    NB.buildClassifier(base);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("Trace from NB Thread.");
                }

                Platform.runLater(() -> finishTaskOfIndex(historyIndex));
            }
        };

        thread.start();
    }

    protected void runRuleBased() throws Exception {
        Thread thread = new Thread(){
            public void run(){
                int historyIndex = historyList.size() - 1;
                M5Rules RB = new M5Rules();

                try {
                    RB.buildClassifier(base);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("Trace from RB Thread.");
                }

                Platform.runLater(() -> finishTaskOfIndex(historyIndex));
            }
        };

        thread.start();
    }

    protected void runNeuralNetwork() throws Exception {
        Thread thread = new Thread(){
            public void run(){
                int historyIndex = historyList.size() - 1;
                MultilayerPerceptron NN = new MultilayerPerceptron();

                try {
                    NN.buildClassifier(base);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("Trace from NN Thread.");
                }

                Platform.runLater(() -> finishTaskOfIndex(historyIndex));
            }
        };

        thread.start();
    }

    protected void runDeepLearning() throws Exception {
//        Thread thread = new Thread(){
//            public void run(){
//                int historyIndex = historyList.size() - 1;;
//                Dl4jMlpClassifier DL = new Dl4jMlpClassifier();
//
//                try {
//                    DL.buildClassifier(base);
//                } catch (Exception e) {
//                    e.printStackTrace();
//                    System.out.println("Trace from DL Thread.");
//                }
//
//                Platform.runLater(() -> finishTaskOfIndex(historyIndex));
//            }
//        };
//
//        thread.start();
    }

    protected void finishTaskOfIndex(int index) {
        historyList.get(index).done.set(true);
    }
}