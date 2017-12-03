package deeplearning.controllers;

import com.jfoenix.controls.*;

import deeplearning.models.HistoryRow;

import javafx.geometry.Pos;
import javafx.scene.layout.HBox;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
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
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.dl4j.updater.Adam;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.File;
import java.net.URL;
import java.util.*;


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
    @FXML private JFXSlider sliderKernelDL;
    @FXML private JFXSlider sliderStrideDL;
    @FXML private JFXTextField textFieldHiddenLayersDL;

    private JFXSlider sliderLearningRateNN;
    private JFXSlider sliderMomentumNN;
    private JFXSlider slideValidationSetSizeNN;
    private JFXTextField textFieldHiddenLayersNN;
    
    private JFXSlider sliderMinBucketSizeRB;

    private Instances base;
    private ObservableList<HistoryRow> historyList;

    private Reorder attOrderFilter = new Reorder();

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        sliderLearningRateNN = new JFXSlider();
        sliderLearningRateNN.setValue(30);

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

                    if(selectedRow.enabledCC && !selectedRow.endedCC.get() && !selectedRow.failCC.get()) {
                        return;
                    }

                    if(selectedRow.enabledDL && !selectedRow.endedDL.get() && !selectedRow.failDL.get()) {
                        return;
                    }

                    if(selectedRow.enabledCC) {
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
        JFXDialog dialog = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

        JFXButton closeButton = new JFXButton("Close".toUpperCase());
        closeButton.setTextFill(Color.WHITE);
        closeButton.setButtonType(JFXButton.ButtonType.RAISED);
        closeButton.setText((row.enabledDL ? "Next" : "Close").toUpperCase());
        closeButton.setStyle(
                "-fx-background-color: #2198F3;\n" +
                        "-fx-end-margin: 10px;\n" + "-fx-start-margin: 10px;\n" + "-fx-spacing: 10px"
        );

        closeButton.setOnAction(event ->  {
            dialog.close();
            if(row.enabledDL) {
                historyViewPresentDLResults(row);
            }
        });

        JFXButton saveButton = new JFXButton("Save".toUpperCase());
        saveButton.setTextFill(Color.WHITE);
        saveButton.setButtonType(JFXButton.ButtonType.RAISED);
        saveButton.setStyle(
                "-fx-background-color: #2198F3;\n"
        );

        saveButton.setOnAction(event -> {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Save CC Results");
            FileChooser.ExtensionFilter txt = new FileChooser.ExtensionFilter("Text files (*.txt)",  "*.txt");
            fileChooser.getExtensionFilters().addAll(txt);

            File f = fileChooser.showSaveDialog(baseInfoClassesLabel.getScene().getWindow());

            try {
                FileUtils.writeStringToFile(f, row.getResultsTextCC());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        ScrollPane scroll = new ScrollPane(new Text(row.getResultsTextCC()));
        scroll.setPadding(new Insets(8, 8,8,8));
        scroll.setFitToWidth(true);

        VBox vbox = new VBox(10);
        vbox.getChildren().add(scroll);

        HBox buttonsHbox = new HBox(10);
        buttonsHbox.setAlignment(Pos.BASELINE_RIGHT);
        buttonsHbox.getChildren().add(saveButton);
        buttonsHbox.getChildren().add(closeButton);
        vbox.getChildren().add(buttonsHbox);

        layout.setBody(vbox);
        layout.setMinWidth(665);

        dialog.show();
    }

    protected void historyViewPresentDLResults(HistoryRow row){
        JFXDialogLayout layout = new JFXDialogLayout();
        JFXDialog dialog = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

        JFXButton closeButton = new JFXButton("Close".toUpperCase());
        closeButton.setTextFill(Color.WHITE);
        closeButton.setStyle("-fx-background-color: #2198F3");
        closeButton.setOnAction(e -> dialog.close());
        closeButton.setButtonType(JFXButton.ButtonType.RAISED);

        JFXButton saveButton = new JFXButton("Save".toUpperCase());
        saveButton.setTextFill(Color.WHITE);
        saveButton.setButtonType(JFXButton.ButtonType.RAISED);
        saveButton.setStyle("-fx-background-color: #2198F3");

        saveButton.setOnAction(event -> {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Save DL Results");
            FileChooser.ExtensionFilter txt = new FileChooser.ExtensionFilter("Text files (*.txt)",  "*.txt");
            fileChooser.getExtensionFilters().addAll(txt);

            File f = fileChooser.showSaveDialog(baseInfoClassesLabel.getScene().getWindow());

            try {
                FileUtils.writeStringToFile(f, row.getResultsTextDL());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        ScrollPane scroll = new ScrollPane(new Text(row.getResultsTextDL()));
        scroll.setPadding(new Insets(8, 8,8,8));
        scroll.setFitToWidth(true);

        VBox vbox = new VBox(10);
        vbox.getChildren().add(scroll);

        HBox buttonsHbox = new HBox(10);
        buttonsHbox.setAlignment(Pos.BASELINE_RIGHT);
        buttonsHbox.getChildren().add(saveButton);
        buttonsHbox.getChildren().add(closeButton);
        vbox.getChildren().add(buttonsHbox);

        layout.setBody(vbox);
        layout.setMinWidth(665);

        dialog.show();
    }

    @FXML protected void handleLoadBase(ActionEvent event) throws Exception {
        File f = loadArff();

        if (f != null && f.getName().endsWith(".arff")) {

            ConverterUtils.DataSource source = new ConverterUtils.DataSource(f.getAbsolutePath());
            base = new Instances(source.getDataSet());

            int[] baseIndexesOrder = new int[base.numAttributes()];

            for(int i = 0; i < base.numAttributes(); i++) {
                baseIndexesOrder[i] = i;
            }

            String baseName = base.relationName();

            attOrderFilter.setAttributeIndicesArray(baseIndexesOrder);
            attOrderFilter.setInputFormat(base);

            base.setClass(base.attribute(base.numAttributes() - 1));

            updateBaseInfoSection();
            base.setRelationName(baseName);
        }
    }

    private Instances formatClass(Instances b) {
        if(base.classAttribute().isNumeric()) {
            return base;
        }

        Instances base = new Instances(b);

        List<String> newClassValeus = new ArrayList<>();
        for(int i = 0; i < base.classAttribute().numValues(); i++) {
            newClassValeus.add(String.valueOf(i));
        }

        String[] oldValues = new String[base.numInstances()];

        for(int i = 0; i < base.numInstances(); i++) {
            oldValues[i] = String.valueOf((int)base.instance(i).value(base.numAttributes()-1));
        }

        Attribute newClass = new Attribute(base.classAttribute().name(), newClassValeus);

        if(base.classIndex() != 0) {
            base.setClassIndex(0);
        } else {
            base.setClassIndex(1);
        }

        base.deleteAttributeAt(base.numAttributes() - 1);
        base.insertAttributeAt(newClass, base.numAttributes());
        base.setClassIndex(base.numAttributes()-1);

        HashMap map = new HashMap();
        int v = 0;

        for(int i = 0; i < base.numInstances(); i++) {
            String classVal = oldValues[i];

            if(!map.containsKey(classVal)) {
                map.put(classVal, String.valueOf(v));
                v++;
            }

            base.instance(i).setClassValue((String) map.get(classVal));
        }

        String className = base.classAttribute().name();
        base.setClassIndex(base.numAttributes() - 1);

        return base;
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
        baseClassComboBox.setValue(options.get(options.size() - 1));
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

        base = Filter.useFilter(base, attOrderFilter);

        for(int i = 0; i < base.numAttributes(); i++) {
            if (base.attribute(i).name() == baseClassComboBox.getValue()) {

                int[] auxAttributeOrder = new int[base.numAttributes()];

                for(int j = 0; j < base.numAttributes(); j++) {
                    auxAttributeOrder[j] = j + 1;
                }

                auxAttributeOrder[auxAttributeOrder.length - 1] = i + 1;
                auxAttributeOrder[i] = auxAttributeOrder.length;

                StringBuilder indexStr = new StringBuilder();
                for(int a = 0; a < auxAttributeOrder.length; a++) {
                    indexStr.append(auxAttributeOrder[a] + ",");
                }

                indexStr.deleteCharAt(indexStr.length() - 1);

                attOrderFilter.setAttributeIndices(indexStr.toString());
                attOrderFilter.setInputFormat(base);

                base.setClassIndex(base.numAttributes() - 1);
                Instances newBase = Filter.useFilter(base, attOrderFilter);
                base = newBase;
                base.setClassIndex(base.numAttributes()-1);

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
            closeButton.setOnAction(e -> dialog.close());
            closeButton.setButtonType(JFXButton.ButtonType.RAISED);

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
            addOption("Learning Rate", sliderLearningRateNN);
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
        Instances b = new Instances(base);

        try {
            J48 c45 = new J48();
            c45.setConfidenceFactor(((float)sliderConfidenceFactorC45.getValue())/100);
            c45.setMinNumObj((int)sliderMinLeaftC45.getValue());

            long t = System.currentTimeMillis();
            c45.buildClassifier(b);
            final long time = System.currentTimeMillis() - t;

            Evaluation e = new Evaluation(b);
            e.crossValidateModel(c45, base, 10, new Random(1));

            Platform.runLater(() -> finishTaskCC(historyIndex, e, c45, time, b.toSummaryString()));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithErrorCC(historyIndex, e));
        }
    }

    protected void runNaiveBayes() throws Exception {
        int historyIndex = historyList.size() - 1;
        Instances b = new Instances(base);

        try {
            NaiveBayes NB = new NaiveBayes();

            long t = System.currentTimeMillis();
            NB.buildClassifier(b);
            final long time = System.currentTimeMillis() - t;

            Evaluation e = new Evaluation(b);
            e.crossValidateModel(NB, b, 10, new Random(1));

            Platform.runLater(() -> finishTaskCC(historyIndex, e, NB, time, b.toSummaryString()));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithErrorCC(historyIndex, e));
        }
    }

    protected void runRuleBased() throws Exception {
        int historyIndex = historyList.size() - 1;
        Instances b = new Instances(base);

        try {
            OneR RB = new OneR();
            RB.setMinBucketSize((int)sliderMinBucketSizeRB.getValue());

            long t = System.currentTimeMillis();
            RB.buildClassifier(b);
            final long time = System.currentTimeMillis() - t;

            Evaluation e = new Evaluation(base);
            e.crossValidateModel(RB, b, 10, new Random(1));

            Platform.runLater(() -> finishTaskCC(historyIndex, e, RB, time, b.toSummaryString()));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithErrorCC(historyIndex, e));
        }
    }

    protected void runNeuralNetwork() throws Exception {
        int historyIndex = historyList.size() - 1;
        Instances b = new Instances(base);
        try {
            MultilayerPerceptron NN = new MultilayerPerceptron();
            NN.setLearningRate(sliderLearningRateNN.getValue()/100);
            NN.setMomentum(sliderMomentumNN.getValue()/100);
            NN.setValidationSetSize((int)slideValidationSetSizeNN.getValue());
            NN.setHiddenLayers(textFieldHiddenLayersNN.getText());

            long t = System.currentTimeMillis();
            NN.buildClassifier(b);
            final long time = System.currentTimeMillis() - t;

            Evaluation e = new Evaluation(b);
            e.crossValidateModel(NN, b, 10, new Random(1));

            Platform.runLater(() -> finishTaskCC(historyIndex, e, NN, time, b.toSummaryString()));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithErrorCC(historyIndex, e));
        }
    }

    protected void runDeepLearning() throws Exception {
        int historyIndex = historyList.size() - 1;
        Instances b = new Instances(base);
        int nAttr = b.numAttributes() - 1;

        try {
            Dl4jMlpClassifier DL = new Dl4jMlpClassifier();
            DL.setSeed(1);
            DL.setNumEpochs(1);

            ConvolutionInstanceIterator iterator = new ConvolutionInstanceIterator();
            iterator.setHeight(1);
            iterator.setNumChannels(1);
            iterator.setWidth(nAttr);

            String[] hiddenLayersParams = textFieldHiddenLayersDL.getText().replaceAll(" ", "").split(",");
            List<Layer> layers = new ArrayList<>();

            for(int i = 0; i < hiddenLayersParams.length; i++) {
                int convLayerSize;

                if(hiddenLayersParams[i].equals("a")) {
                    convLayerSize = (nAttr + b.numClasses())/2;
                } else if(hiddenLayersParams[i].equals("i")) {
                    convLayerSize = nAttr;
                } else if(hiddenLayersParams[i].equals("o")) {
                    convLayerSize = b.numClasses();
                } else {
                    convLayerSize = Integer.parseInt(hiddenLayersParams[i]);
                }

                ConvolutionLayer convLayer = new ConvolutionLayer();
                convLayer.setNOut(convLayerSize);
                convLayer.setConvolutionMode(ConvolutionMode.Same);
                convLayer.setActivationFn(new ActivationReLU());
                convLayer.setKernelSize(new int[]{1, (int) sliderKernelDL.getValue()});
                convLayer.setStride(new int[]{(int) sliderStrideDL.getValue(), 1});
                convLayer.setLayerName("CONV " + i);


                SubsamplingLayer subSamplignLayer = new SubsamplingLayer();
                subSamplignLayer.setPoolingType(PoolingType.MAX);
                subSamplignLayer.setKernelSize(new int[]{1, 2});
                subSamplignLayer.setStride(new int[]{2, 1});
                subSamplignLayer.setLayerName("POOLING " + i);

                layers.add(convLayer);
                layers.add(subSamplignLayer);
            }

            OutputLayer outputLayer = new OutputLayer();
            outputLayer.setActivationFn(new ActivationSoftmax());
            outputLayer.setLossFn(new LossMCXENT());
            outputLayer.setNOut(b.numClasses());
            outputLayer.setLayerName("OUTPUT");
            layers.add(outputLayer);

            NeuralNetConfiguration nnc = new NeuralNetConfiguration();
            nnc.setUpdater(new Adam());
            nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
            nnc.setLearningRate(sliderLearningRateDL.getValue()/100);

            DL.setNeuralNetConfiguration(nnc);

            DL.setInstanceIterator(iterator);

            Layer[] convNeuralLayers = new Layer[hiddenLayersParams.length * 2 + 1];
            layers.toArray(convNeuralLayers);
            DL.setLayers(convNeuralLayers);

            long t = System.currentTimeMillis();
            DL.buildClassifier(b);
            final long time = System.currentTimeMillis() - t;

            Evaluation e = new Evaluation(b);
            e.crossValidateModel(DL, b, 10, new Random(1));

            Platform.runLater(() -> finishTaskDL(historyIndex, e, DL, time, b.toSummaryString()));
        } catch(Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithErrorDL(historyIndex,e));
        }
    }

    protected void finishTaskCC(int index, Evaluation e, Classifier c, long t, String baseInfo) {
        historyList.get(index).classifierCC = c;
        historyList.get(index).evalCC = e;
        historyList.get(index).elapsedTimeCC = t;
        historyList.get(index).baseInfoCC = baseInfo;
        historyList.get(index).endedCC.set(true);
    }

    protected void finishTaskDL(int index, Evaluation e, Classifier c, long t, String baseInfo) {
        historyList.get(index).classifierDL = c;
        historyList.get(index).evalDL = e;
        historyList.get(index).elapsedTimeDL = t;
        historyList.get(index).baseInfoDL = baseInfo;
        historyList.get(index).endedDL.set(true);
    }

    protected void finishTaskOfIndexWithErrorCC(int index, Exception e) {
        historyList.get(index).exceptionCC = e;
        historyList.get(index).failCC.set(true);
    }

    protected void finishTaskOfIndexWithErrorDL(int index, Exception e) {
        historyList.get(index).exceptionDL = e;
        historyList.get(index).failDL.set(true);
    }
}