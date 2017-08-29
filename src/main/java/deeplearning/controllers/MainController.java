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
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
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
import java.util.Random;
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

    @FXML private VBox parametersBox;

    @FXML private JFXSlider sliderConfidenceFactorC45;
    @FXML private JFXSlider sliderMinLeaftC45;

    private JFXSlider slideLearningrRateNN;
    private JFXSlider slideMomentumNN;
    private JFXSlider slideValidationSetSizeNN;
    private JFXTextField textFieldHiddenLayersNN;
    
    private JFXSlider sliderMinBucketSizeRB;

    private Instances base;
    private ObservableList<HistoryRow> historyList;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        baseClassComboBox.setVisibleRowCount(7);

        historyList = FXCollections.observableArrayList(HistoryRow.extractor());
        historyListView.setItems(historyList);
        historyListView.setCellFactory(studentListView -> new HistoryListViewCell());

        slideLearningrRateNN = new JFXSlider();
        slideMomentumNN = new JFXSlider();
        slideValidationSetSizeNN = new JFXSlider();
        textFieldHiddenLayersNN = new JFXTextField();
        textFieldHiddenLayersNN.setUnFocusColor(Color.WHITE);

        sliderMinBucketSizeRB = new JFXSlider();
        sliderMinBucketSizeRB.setMax(10);
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
                addToHistory(builder.toString(), checkBoxEnableDL.isSelected(), checkBoxEnableCC.isSelected());
                runC45();

            } else if (radioButtonNaiveBayes.isSelected()) {
                builder.append("NB");
                addToHistory(builder.toString(), checkBoxEnableDL.isSelected(), checkBoxEnableCC.isSelected());

            } else if (radioButtonNeuralNetwork.isSelected()) {
                builder.append("NN");
                addToHistory(builder.toString(), checkBoxEnableDL.isSelected(), checkBoxEnableCC.isSelected());
                runNeuralNetwork();

            } else if (radioButtonRuleBased.isSelected()) {
                builder.append("RB");
                addToHistory(builder.toString(), checkBoxEnableDL.isSelected(), checkBoxEnableCC.isSelected());
                runRuleBased();
            }
        }

        if (checkBoxEnableDL.isSelected()) {
            if(!checkBoxEnableCC.isSelected()) {
                builder.append("DL");
                addToHistory(builder.toString(), checkBoxEnableDL.isSelected(), checkBoxEnableCC.isSelected());
            }

            runDeepLearning();
        }
    }

    @FXML protected void updateClassifierOptions(ActionEvent event) throws Exception {
        parametersBox.getChildren().clear();

        if(radioButtonC45.isSelected()) {
            addOption("Confidence Factor", sliderConfidenceFactorC45);
            addOption("Min. intances per leaf", sliderMinLeaftC45);
        } else if(radioButtonNaiveBayes.isSelected()) {
            addOption("No Parameters", null);
        } else if(radioButtonNeuralNetwork.isSelected()) {
            addOption("Learning Rate", slideLearningrRateNN);
            addOption("Momentum", slideMomentumNN);
            addOption("Validation Set Size", slideValidationSetSizeNN);
            addOption("Hidden Layers", textFieldHiddenLayersNN);
        } else if(radioButtonRuleBased.isSelected()) {
            addOption("Min. Bucket Size", sliderMinBucketSizeRB);
        }
    }

    protected void addOption(String title, Node n) {
        Label titleLabel = new Label(title);

        titleLabel.setFont(new Font("Roboto", 15));
        titleLabel.setTextFill(Color.WHITE);

        parametersBox.getChildren().add(titleLabel);
        if(n != null) parametersBox.getChildren().add(n);
    }

    protected void addToHistory(String title, boolean isRunningDL, boolean isRunningCC) {
        StringBuilder builder = new StringBuilder(title);
        DateFormat dateFormat = new SimpleDateFormat(" @ HH:mm:ss");
        builder.append(dateFormat.format(new Date()));

        historyList.add(new HistoryRow(builder.toString(), isRunningDL, isRunningCC));
    }


    protected void runC45() throws Exception{
        new Thread(() -> {
            int historyIndex = historyList.size() - 1;
            try {
                J48 c45 = new J48();
                c45.setConfidenceFactor(((float)sliderConfidenceFactorC45.getValue())/100);
                c45.setMinNumObj((int)sliderMinLeaftC45.getValue());
                c45.buildClassifier(base);
                Evaluation e = new Evaluation(base);
                e.crossValidateModel(c45, base, 10, new Random(new Date().getTime()));
                Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e));
            } catch (Exception e) {
                e.printStackTrace();
                Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
            }
        }).start();
    }

    protected void runNaiveBayes() throws Exception {
        new Thread(() -> {

            int historyIndex = historyList.size() - 1;
            try {
                NaiveBayes NB = new NaiveBayes();
                NB.buildClassifier(base);

                Evaluation e = new Evaluation(base);
                e.crossValidateModel(NB, base, 10, new Random(new Date().getTime()));
                Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e));
            } catch (Exception e) {
                e.printStackTrace();
                Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
            }
        }).start();
    }

    protected void runRuleBased() throws Exception {
        new Thread(() -> {
            int historyIndex = historyList.size() - 1;

            try {
                OneR RB = new OneR();
                RB.setMinBucketSize((int)sliderMinBucketSizeRB.getValue());
                RB.buildClassifier(base);

                Evaluation e = new Evaluation(base);
                e.crossValidateModel(RB, base, 10, new Random(new Date().getTime()));
                Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e));
            } catch (Exception e) {
                e.printStackTrace();
                Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
            }
        }).start();
    }

    protected void runNeuralNetwork() throws Exception {
        new Thread(() -> {
            int historyIndex = historyList.size() - 1;
                try {
                    MultilayerPerceptron NN = new MultilayerPerceptron();
                    NN.setLearningRate(slideLearningrRateNN.getValue());
                    NN.setMomentum(slideMomentumNN.getValue());
                    NN.setValidationSetSize((int)slideValidationSetSizeNN.getValue());
                    NN.setHiddenLayers(textFieldHiddenLayersNN.getText());
                    NN.buildClassifier(base);

                    Evaluation e = new Evaluation(base);
                    e.crossValidateModel(NN, base, 10, new Random(new Date().getTime()));
                    Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e));
                } catch (Exception e) {
                    e.printStackTrace();
                    Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
                }
        }).start();
    }

    protected void runDeepLearning() throws Exception {
        new Thread(() -> {
            int historyIndex = historyList.size() - 1;

            try {
                Platform.runLater(() -> finishDLTaskOfIndex(historyIndex, null));
            } catch(Exception e) {
                Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
            }
        }).start();
    }

    protected void finishDLTaskOfIndex(int index, Evaluation eval) {
        historyList.get(index).endedDL.set(true);
        historyList.get(index).evalDL.set(eval);
    }

    protected void finishCCTaskOfIndex(int index, Evaluation eval) {
        historyList.get(index).endedCC.set(true);
        historyList.get(index).evalCC.set(eval);
    }

    protected void finishTaskOfIndexWithError(int index) {
        historyList.get(index).fail.set(true);
    }
}