package deeplearning.controllers;

import com.jfoenix.controls.*;

import com.sun.javafx.collections.ImmutableObservableList;
import com.sun.javafx.collections.ObservableListWrapper;
import deeplearning.models.HistoryRow;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.ObservableListBase;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.activations.IActivation;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.rules.OneR;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.dl4j.activations.ActivationIdentity;
import weka.dl4j.layers.BatchNormalization;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.SubsamplingLayer;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
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

    @FXML private JFXSlider sliderLearningRateDL;
    @FXML private JFXSlider sliderMomentumDL;
    @FXML private JFXSlider sliderRegularizationConstantDL;
    @FXML private JFXSlider sliderNumberOfConvLayersDL;

    @FXML private JFXSlider sliderKernelXDL;
    @FXML private JFXSlider sliderKernelYDL;

    @FXML private JFXSlider sliderNumberOfEpochsDL;

    @FXML private JFXComboBox poolingTypeComboBox;

    private JFXSlider sliderLearningrRateNN;
    private JFXSlider sliderMomentumNN;
    private JFXSlider slideValidationSetSizeNN;
    private JFXTextField textFieldHiddenLayersNN;
    
    private JFXSlider sliderMinBucketSizeRB;

    private Instances base;
    private ObservableList<HistoryRow> historyList;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        baseClassComboBox.setVisibleRowCount(7);

        ObservableList<String> pooling = new ImmutableObservableList<>("MAX","SUM","AVG","PNORM");
        poolingTypeComboBox.setItems(pooling);
        poolingTypeComboBox.setValue("MAX");

        sliderLearningrRateNN = new JFXSlider();
        sliderLearningrRateNN.setValue(30);


        sliderMomentumNN = new JFXSlider();
        sliderMomentumNN.setValue(20);
        slideValidationSetSizeNN = new JFXSlider();

        textFieldHiddenLayersNN = new JFXTextField();
        textFieldHiddenLayersNN.setUnFocusColor(Color.WHITE);
        textFieldHiddenLayersNN.setText("a");

        sliderMinBucketSizeRB = new JFXSlider();
        sliderMinBucketSizeRB.setMax(10);
        sliderMinBucketSizeRB.setValue(6);

        historyList = FXCollections.observableArrayList(HistoryRow.extractor());
        historyListView.setItems(historyList);

        historyListView.setCellFactory(studentListView -> {
            HistoryListViewCell cell = new HistoryListViewCell();

            cell.setOnMouseClicked(e -> {
                cell.updateSelected(false);

                if(e.getClickCount() == 2) {
                    HistoryRow selectedRow = historyListView.getSelectionModel().getSelectedItems().get(0);

                    if(selectedRow.isRunningCC && !selectedRow.endedCC.get()) {
                        return;
                    }

                    if(selectedRow.isRunningDL && !selectedRow.endedDL.get()) {
                        return;
                    }

                    if(selectedRow.isRunningCC) {
                        historyViewPresentCCResults(selectedRow);
                    } else {
                        historyViewPresentDLResults(selectedRow);
                    }
                }
            });

            return cell;
        });
    }

    protected void historyViewPresentCCResults(HistoryRow row){


        JFXDialogLayout layout = new JFXDialogLayout();
        ScrollPane scroll = new ScrollPane(new Text(row.getResultsTextCC()));
        scroll.setPadding(new Insets(8, 8,8,8));
        scroll.setFitToWidth(true);

        layout.setBody(scroll);
        layout.setMinWidth(665);

        JFXDialog dialog = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

        String buttonTitle;
        EventHandler<ActionEvent> buttonHandler;

        if(row.isRunningDL) {
            buttonTitle = "See Deep Learning Results";

            buttonHandler = new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent event) {
                    JFXDialogLayout layout = new JFXDialogLayout();

                    ScrollPane scroll = new ScrollPane(new Text(row.getResultsTextDL()));
                    scroll.setPadding(new Insets(8, 8,8,8));
                    scroll.setFitToWidth(true);

                    layout.setBody(scroll);
                    layout.setMinWidth(665);

                    JFXDialog dialogDL = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

                    JFXButton closeButton = new JFXButton("Close".toUpperCase());
                    closeButton.setTextFill(Color.WHITE);
                    closeButton.setStyle("-fx-background-color: #2198F3");
                    closeButton.setOnAction(new EventHandler<ActionEvent>() {
                        @Override
                        public void handle(ActionEvent event) {
                            dialogDL.close();
                        }
                    });

                    layout.setActions(closeButton);
                    dialog.close();
                    dialogDL.show();
                }
            };
        } else {
            buttonTitle = "Close";
            buttonHandler = new EventHandler<ActionEvent>() {

                @Override
                public void handle(ActionEvent event) {
                    dialog.close();
                }
            };
        }

        JFXButton closeButton = new JFXButton(buttonTitle.toUpperCase());
        closeButton.setTextFill(Color.WHITE);
        closeButton.setStyle("-fx-background-color: #2198F3");
        closeButton.setOnAction(buttonHandler);
        layout.setActions(closeButton);

        dialog.show();
    }

    protected void historyViewPresentDLResults(HistoryRow row){

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

        boolean showDialog = true;
        JFXDialogLayout layout = new JFXDialogLayout();

        if(base == null) {
            layout.setBody(new Text("Load a base file before you run." +
                    "\nYou can load a base file with the button on the top-left."));
        } else if (!checkBoxEnableDL.isSelected() && !checkBoxEnableCC.isSelected()) {
            layout.setBody(new Text("Select a classification training method to use." +
                    "\nEnable at least one with the checkbox on either\nDEEP LEARNING or CLASSIC CLASSFICATION."));
        } else {
            showDialog = false;
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

        HistoryRow r = new HistoryRow(checkBoxEnableDL.isSelected(), checkBoxEnableCC.isSelected());

        Thread ccThread = new Thread();

        if (checkBoxEnableCC.isSelected()) {
            if (radioButtonC45.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runC45();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.C45;
            } else if (radioButtonNaiveBayes.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runNaiveBayes();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.NB;
            } else if (radioButtonNeuralNetwork.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runNeuralNetwork();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.NN;
            } else if (radioButtonRuleBased.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runRuleBased();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.RB;
            }
        }

        Thread dlThread = new Thread();


        if (checkBoxEnableDL.isSelected()) {
            dlThread = new Thread(() -> {
                try {
                    runDeepLearning();
                } catch(Exception e) {
                    e.printStackTrace();
                }
            });
        }

        historyList.add(r);
        dlThread.start();
        ccThread.start();
    }

    @FXML protected void updateClassifierOptions(ActionEvent event) throws Exception {
        parametersBox.getChildren().clear();

        if(radioButtonC45.isSelected()) {
            addOption("Confidence Factor", sliderConfidenceFactorC45);
            addOption("Min. intances per leaf", sliderMinLeaftC45);
        } else if(radioButtonNaiveBayes.isSelected()) {
            addOption("No Parameters", null);
        } else if(radioButtonNeuralNetwork.isSelected()) {
            addOption("Learning Rate", sliderLearningrRateNN);
            addOption("Momentum", sliderMomentumNN);
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

    protected void runC45() throws Exception{
        int historyIndex = historyList.size() - 1;

        try {
            J48 c45 = new J48();
            c45.setConfidenceFactor(((float)sliderConfidenceFactorC45.getValue())/100);
            c45.setMinNumObj((int)sliderMinLeaftC45.getValue());
            c45.buildClassifier(base);
            Evaluation e = new Evaluation(base);
            e.crossValidateModel(c45, base, 10, new Random(1));
            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, c45));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runNaiveBayes() throws Exception {
        int historyIndex = historyList.size() - 1;
        try {
            NaiveBayes NB = new NaiveBayes();
            NB.buildClassifier(base);
            Evaluation e = new Evaluation(base);
            e.crossValidateModel(NB, base, 10, new Random(1));
            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, NB));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runRuleBased() throws Exception {
        int historyIndex = historyList.size() - 1;

        try {
            OneR RB = new OneR();
            RB.setMinBucketSize((int)sliderMinBucketSizeRB.getValue());
            RB.buildClassifier(base);

            Evaluation e = new Evaluation(base);
            e.crossValidateModel(RB, base, 10, new Random(1));
            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, RB));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runNeuralNetwork() throws Exception {
        int historyIndex = historyList.size() - 1;
        try {
            MultilayerPerceptron NN = new MultilayerPerceptron();
            NN.setLearningRate(sliderLearningrRateNN.getValue()/100);
            NN.setMomentum(sliderMomentumNN.getValue()/100);
            NN.setValidationSetSize((int)slideValidationSetSizeNN.getValue());
            NN.setHiddenLayers(textFieldHiddenLayersNN.getText());
            NN.buildClassifier(base);

            Evaluation e = new Evaluation(base);
            e.crossValidateModel(NN, base, 10, new Random(1));
            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, NN));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runDeepLearning() throws Exception {
        int historyIndex = historyList.size() - 1;

        try {
            Dl4jMlpClassifier DL = new Dl4jMlpClassifier();

            int nLayers = (int) sliderNumberOfConvLayersDL.getValue();
            ArrayList<Layer> layers = new ArrayList<Layer>();

            for(int i = 0; i < nLayers; i++) {
                ConvolutionLayer l = new ConvolutionLayer();
                l.setLearningRate(sliderLearningRateDL.getValue()/100);
                l.setMomentum(sliderMomentumDL.getValue()/100);
                l.setKernelSizeX((int)sliderKernelXDL.getValue());
                l.setKernelSizeX((int)sliderKernelYDL.getValue());
                l.setStrideX(2);
                l.setStrideY(2);
                l.setPaddingX(2);
                l.setPaddingY(2);
                l.setLayerName(i + " Conv Layer");

                SubsamplingLayer sb = new SubsamplingLayer();
                sb.setStrideX(2);
                sb.setStrideY(2);
                sb.setKernelSizeX(2);
                sb.setKernelSizeY(2);
                sb.setLayerName(i + "Pooling Layer");
                sb.setActivationFn(new ActivationIdentity());
                sb.setActivationFunction(new org.nd4j.linalg.activations.impl.ActivationIdentity());

                System.out.println(poolingTypeComboBox.getValue());

                switch((String)poolingTypeComboBox.getValue()) {
                    case "MIN":
                        sb.setPoolingType(PoolingType.SUM);
                        break;

                    case "AVG":
                        sb.setPoolingType(PoolingType.AVG);
                        break;

                    case "PNORM":
                        sb.setPoolingType(PoolingType.PNORM);
                        break;

                    case "MAX":
                        sb.setPoolingType(PoolingType.MAX);
                        break;
                }

                layers.add(l);
                layers.add(sb);
            }

            DL.setLayers((Layer[]) layers.toArray());
            DL.setNumEpochs((int)sliderNumberOfEpochsDL.getValue());

            DL.buildClassifier(base);
            Evaluation e = new Evaluation(base);
            e.crossValidateModel(DL, base, 10, new Random(1));
            Platform.runLater(() -> finishDLTaskOfIndex(historyIndex, e, DL));
        } catch(Exception _) {
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }


    protected void finishDLTaskOfIndex(int index, Evaluation eval, Classifier c) {
        historyList.get(index).endedDL.set(true);
        historyList.get(index).evalDL = eval;
        historyList.get(index).classifierDL = c;
    }

    protected void finishCCTaskOfIndex(int index, Evaluation eval, Classifier c) {
        historyList.get(index).endedCC.set(true);
        historyList.get(index).evalCC = eval;
        historyList.get(index).classifierCC = c;
    }

    protected void finishTaskOfIndexWithError(int index) {
        historyList.get(index).fail.set(true);
    }
}