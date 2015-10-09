%%  EEL 6825 Jose C Fernandez
% MAMMOGRAPHIC MASSES CLASSIFICATION PROJECT
%GROUP 19
%% Note
% All Graphs are displayed fullscreen for clarity
% there is a text report generated with results and conclusions
% on the main screen

%% References
% All functions and examples form Matlab Help and Matlab Doc Copyright of
% The MathWorks Inc 1984-2010
% Dr. Seth McNeill Lectures and  examples EEL 6825
% http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries
% Introduction to image processing in Matlab 1 by Kristian Sandberg, Department of Applied Mathematics, University of Colorado at Boulder
% Basic Concepts in Matlab Michael G. Kay Fitts Dept. of Industrial and Systems Engineering North Carolina State University
%  http://www.data-compression.com/index.shtml
% http://www.ifp.uiuc.edu/~minhdo/teaching/speaker_recognition
% BI-RADS Classification for Management of Abnormal Mammograms
% http://www.jabfm.org/content/19/2/161/T1.expansion.html

%% Clean up
clear
close all
clc
%% Acquiring Data
% UCI data
% Attribute Information:
 %  1. BI-RADS assessment: 1 to 5 (ordinal)
 %  2. Age: patient's age in years (integer)
 %  3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
 %  4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
 %  5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
 %  6. Severity: benign=0 or malignant=1 (binominal)

% Reading the data from UCI webpage, stored in source_data

% source_data = urlread('http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data');
% Working version
source_data = urlread('http://mlearn.ics.uci.edu/databases/mammographic-masses/mammographic_masses.data');
% fileId = fopen('mammographic_masses.data.txt','r');
% formatSpec = '%f';
% tmp = textscan(fileId, '%s','Delimiter', '\n');
% fclose(fileId);

% source_data = regex(tmp,'','split');
% source_data = ca(1,source_data{:});



% Substituting the unknown fields by empty ''
data = regexprep(source_data, '?', '-1');

 % Convertig string data to numeric data
 numeric_data=str2num(data);
% Declaring attribute fields

BIRADS = numeric_data(:,1);
Age=numeric_data(:,2);
Shape=numeric_data(:,3);
Margin=numeric_data(:,4);
Density=numeric_data(:,5);
Severity=numeric_data(:,6);

% source_data is a matrix of 6 columns contining patient information
source_data = [BIRADS Age Shape Margin Density Severity ];

%% Data treatment by  Listwise Deletion (Complete case analysis. )

%I am using listwise reduction to eliminate patients with 1
% missing data. That way I will achieve unbiased parameter estimates.

%  source_data_with_all_parameters is a matrix of 6 columns contining patient information without
% any missing parameters
% source_data(source_data~=-1);


n=size(source_data,1);

c=0;
for ii=1:n
 if source_data(ii,:)~=-1;
  c=c+1;
  source_data_with_all_parameters(c,:)= source_data(ii,:);
end

end

% Since the data ranges are known parameters we can eliminate outliers
nn=size(source_data_with_all_parameters(:,1),1);

cc=0;
for jj=1:nn
    if ((source_data_with_all_parameters (jj,1)>=1 && source_data_with_all_parameters(jj,1)<=5) &&...  %  1. BI-RADS assessment: 1 to 5 (ordinal)
        (source_data_with_all_parameters (jj,2)>=15 && source_data_with_all_parameters(jj,2)<=95 )&&...  %  2. Age: patient's age in years (integer)
        (source_data_with_all_parameters (jj,3)>=1 && source_data_with_all_parameters(jj,3)<=4 )&&...  %  3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
        (source_data_with_all_parameters (jj,4)>=1 && source_data_with_all_parameters(jj,5)<=4 )&&...  %  4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
        (source_data_with_all_parameters (jj,5)>=1 && source_data_with_all_parameters(jj,4)<=5 )&&...  %  5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
        (source_data_with_all_parameters (jj,6)>=0 && source_data_with_all_parameters(jj,6)<=1) )      %  6. Severity: benign=0 or malignant=1 (binominal)

       cc=cc+1;
       fully_sanitized_data(cc,:)= source_data_with_all_parameters(jj,:);
      %  fully_sanitized_data is a matrix of 6 columns contining patient information without
      % any missing parameters and without outliers
    end

  end

%% Divide the dataset into training and test sets

% Training Set   40% of Data

whole_set_size=size(fully_sanitized_data(:,1),1); %Size of the whole set

training_set=floor((whole_set_size)*.4);  % 40 % of the data
testing_set=floor((whole_set_size)*.6);  % 60 % of the data

training_set_random_indexes=randi([1 whole_set_size],training_set,1);% random indexes of 40% of the data
testing_set_random_indexes=randi([1 whole_set_size],testing_set,1);% random indexes of 60% of the data

randomized_training_set=fully_sanitized_data(training_set_random_indexes ,1:6);
randomized_testing_set=fully_sanitized_data(testing_set_random_indexes ,1:6);

%disp('Data OK!')

%% Definition of malign and benign classifier matrices

  benign_patient_shape_parameters=fully_sanitized_data( fully_sanitized_data(:,3)<3,3); % Matrix Containing the shape parameters for benign patients
  benign_patient_margin_parameters=fully_sanitized_data( fully_sanitized_data(:,4)<=3,4); % Matrix Containing the margin parameters for benign patients
  benign_patient_density_parameters=fully_sanitized_data(fully_sanitized_data(:,5 )>=3,5 ); % Matrix Containing the density parameters for benign patients
  size_training_set_benign=1:60;
  benign_training_set=[benign_patient_shape_parameters(size_training_set_benign,1) , benign_patient_margin_parameters(size_training_set_benign,1), benign_patient_density_parameters(size_training_set_benign,1) ];
  benign_training_set=unique(benign_training_set,'rows');
  size(benign_training_set,1);

  malign_patient_shape_parameters=fully_sanitized_data( fully_sanitized_data(:,3)>=3,3); % Matrix Containing the shape parameters for malign patients
  malign_patient_margin_parameters=fully_sanitized_data( fully_sanitized_data(:,4)>3,4); % Matrix Containing the margin parameters for malign patients
  malign_patient_density_parameters=fully_sanitized_data(fully_sanitized_data(:,5 )<3,5 ); % Matrix Containing the density parameters for malign patients
  size_training_set_benign=1:60;

  malign_patient_training_set=[malign_patient_shape_parameters(size_training_set_benign,1) , malign_patient_margin_parameters(size_training_set_benign,1), malign_patient_density_parameters(size_training_set_benign,1) ];
  malign_patient_training_set=unique(malign_patient_training_set,'rows');
  size(malign_patient_training_set,1);


  %
  malign_patient_shape_parameters=fully_sanitized_data( fully_sanitized_data(:,3)>=3,3);
  malign_patient_margin_parameters=fully_sanitized_data( fully_sanitized_data(:,4)>=4,4);
  malign_patient_density_parameters=fully_sanitized_data(fully_sanitized_data(:,5 )<= 2,5 );


 %  generates the training for bening
 training2=[ 1 1 3 ];
 for sh=1:4
  for mg=1:5
    for ds=1:4
      if sh<3 && mg<=3 && ds>=3
        training1=[sh mg ds];
        training2=cat(1,training2,training1);
      end

    end
  end
end
training3=training2(2:end,:);
trainingbenign=training3;

  %  generates the training for malign
  training4=[ 3 4 1 ];
  for sh=1:4
    for mg=1:5
      for ds=1:4
        if sh>=3 && mg>3 && ds<3
          training5=[sh mg ds];
          training4=cat(1,training4,training5);
        end

      end
    end
  end
  training6=training4(2:end,:) ;

  trainingmalign=training6;


  training=cat(1, training3, training6);



  group0=0;
  group0=repmat(group0, [ size(training3,1) , 1 ]);
  group1=1;
  group1=repmat(group1, [ size(training6,1) , 1 ]);

 group=cat(1, group0, group1); % This grouping variable will be used for the
 % classify and KNNclassify Matlab functions in order to classify the data.




  %% Creating the risk MDdiagnostic classify as malignant by Doctors severity
  % attribute

  h=0; Group6=zeros(10,6);
  for j=1:length(fully_sanitized_data);

    if fully_sanitized_data(j,6)==1
     h=h+1;
     Group6(h,:)= fully_sanitized_data(j,:);
   end;

 end

 MDdiagnostic=size(Group6,1);

 % Group6 will contain all the patients already classified by the Doctors as
 % malignant. It will be used to compare the classifiers performance


%% Plot of  Parameters Distribution
scrsz = get(0,'ScreenSize');
figure('Name',' Parameters Distribution',     'NumberTitle','off','Position',[1 scrsz(4)  scrsz(3) scrsz(4)])



subplot(2,2,[1 3])
hist( fully_sanitized_data(:,3:5))
hold all

legend(' Shape ', 'Margin', 'Density','Location','SouthOutside', ...
  'Orientation', 'horizontal');
title('Parameters Distribution ','fontsize',25)

 % Formatting Axis and Legends

 xlabel('Parameters','fontsize',25);

 ylabel('Patient ','fontsize',25);

 subplot(2,2,2)
 hist( fully_sanitized_data(:,2))
 legend('Age Distribution','Location','SouthOutside', ...
  'Orientation', 'horizontal');

 title('Age Distribution ','fontsize',25)

 % Formatting Axis and Legends

 xlabel('Age Distribution','fontsize',25);

 ylabel('Patient ','fontsize',25);


 subplot(2,2,4)

 hist( fully_sanitized_data(:,1))
 legend('BIRADS Distribution','Location','SouthOutside', ...
  'Orientation', 'horizontal');
 title('BIRADS Distribution ','fontsize',25)

 % Formatting Axis and Legends

 xlabel('BIRADS ','fontsize',25);

 ylabel('Patient ','fontsize',25);


 %% KNN

%Class = knnclassify(Sample, Training, Group) classifies the rows of the
%data matrix Sample into groups, based on the grouping of the rows of
%Training. Sample and Training must be matrices with the same number of
%columns. Group is a vector whose distinct values define the grouping of
%the rows in Training. Each row of Training belongs to the group whose
%value is the corresponding entry of Group. knnclassify assigns each row
%of Sample to the group for the closest row of Training. Group can be a
%numeric vector, a string array, or a cell array of strings. Training and
%Group must have the same number of rows. knnclassify treats NaNs or empty
%strings in Group as missing values, and ignores the corresponding rows of
%Training. Class indicates which group each row of Sample has been assigned
%to, and is of the same type as Group
 %----------------------------------
 % KNN for Training Data 40%
knn_iterations=20; % Amount of iterations performed by KNN

Fullsample40=[ randomized_training_set(:,3)    randomized_training_set(:,4)   randomized_training_set(:,5)];

clear k idx
for k=1:knn_iterations


   % Classifiying k clusters
   Class40 = knnclassify(Fullsample40, training, group,k);

% Error for Shape----------------

nWrong40=sum(Class40(:)~= randomized_training_set(:,3));
pctWrong40=nWrong40/size(fully_sanitized_data,1)*100;

 %  disp(['Percentage  wrong for Shape KNN ',num2str(pctWrongKS),' %'])
 pctright40=100-pctWrong40;
%disp('')
%disp(['Percentage  right for  KNN ',num2str(pctright40),' %'])
mm(k)=pctright40;


end


% Determine the best performance for KNN
[m,idx] = max(mm, [], 2);


% Since I already determined my best k, I'm going to classify using it and
% form my risk Group 1 using it.
clear Class40
Class40 = knnclassify(Fullsample40, training, group,idx) ;

Knnmalign40=size(Class40(Class40==1),1);  % This is how many patiente classify as malignant




pmd=100*MDdiagnostic/size(fully_sanitized_data,1);
pknn40=100*Knnmalign40/size(fully_sanitized_data,1);
comp40=pmd-pknn40;

 % Display results KNN

 disp('----------------  Results  ---------------')


 disp(['There are   ',num2str(MDdiagnostic),' patients'])
 disp('with probable malignant diagnostic ')
 disp('according to the Doctors.')
 disp('    ')
 disp('-----------------   ooo   --------------')
 disp('    ')

 disp('   ')
 disp(['The best performance for KNN Classification'])
 disp(['using the Training Set  was  ',num2str(100-comp40),' % right with ',num2str(idx), ' iteration(s)'])
 disp(['out of ', num2str(knn_iterations),' iterations'])

% disp('    KNN for the Training set ( 40% )   ')
% disp(['There are   ',num2str(Knnmalign40),' patients'])
% disp('with probable malignant diagnostic ')
% disp('according to the KNN for the training set')

  %disp('    ')
 %disp(['The KNN classifier diagnostic for the training set differs  ',num2str(100-comp40),' % '])
 %disp('of patients with probable malignant diagnostic from ')
 %disp('the Doctors diagnostic')
 disp('----------------------------------')



 % set up the domain over which you want to visualize the decision
% boundary

ns = size(randomized_training_set(:,3),1);


xrange = [1 ns];
yrange = [1 5];
% step size for how finely you want to visualize the decision boundary.
incr = 1;

indexes_patient_amount_training_set_lda=1:ns;

% Define  some training data for three classes.
train40 = cell(3,1);
train40{1} =[randomized_training_set(:,3) indexes_patient_amount_training_set_lda'];
train40{2} =[randomized_training_set(:,4)  indexes_patient_amount_training_set_lda'];
train40{3} = [randomized_training_set(:,5) indexes_patient_amount_training_set_lda'] ;



%% Show the image KNN Training Set
scrsz = get(0,'ScreenSize');

figure('Name','KNN Classifier for the Training Set',...
 'NumberTitle','off','Position',[1 scrsz(4) scrsz(3) scrsz(4)])
hold all

hold on;
set(gca,'ydir','normal');


 %plot the class training data.
 plot(train40{1}(:,2),train40{1}(:,1), 'c.','MarkerSize',10);
 plot(train40{2}(:,2),train40{2}(:,1), 'g.','MarkerSize',10);
 plot(train40{3}(:,2),train40{3}(:,1), 'm.','MarkerSize',10);
% plot the classification
plot(indexes_patient_amount_training_set_lda(Class40==0),Fullsample40(Class40==0),'ob','LineWidth',.2,'MarkerSize',20);
plot(indexes_patient_amount_training_set_lda(Class40==1),Fullsample40(Class40==1),'or','LineWidth',.2,'MarkerSize',20);

  % Formatting Axis and Legends

  xlabel(['Patient by number.  ' num2str( max(ns) ) ' patients displayed'] ,'FontSize',25)
  ylabel('  Parameters Shape/Margin/Density ','FontSize',25)
  title('KNN Classifier for the Training Set','FontSize',25)
  title({'';[...
    'KNN Classifier for the  Training Set classified  successfully  ',num2str(m)...
    ' % of the patients ']},'fontsize',16)

  le = legend('Shape','Margin','Density','Patients Classified as Benign',...
   'Patients Classified as Malign','location','North', 'orientation', 'horizontal');
  set(le,'Interpreter','none')

% End of KNN Training Set



%%  KNN for Test Data 60%
clear m idx mm

knn_iterations=20; % Amount of iterations performed by KNN

full_sample_testing_set=[ randomized_testing_set(:,3)    randomized_testing_set(:,4)   randomized_testing_set(:,5)];

clear k idx
for k=1:knn_iterations


   % Classifying k clusters
   k_cluster_classification_testing_set = knnclassify(full_sample_testing_set, training, group,k);

% Error for Shape----------------

incorrect_amount_testing_set=sum(k_cluster_classification_testing_set(:)~= randomized_testing_set(:,3));
percentage_wrong_testing_set=incorrect_amount_testing_set/size(fully_sanitized_data,1)*100;

 %  disp(['Percentage  wrong for Shape KNN ',num2str(pctWrongKS),' %'])
 percentage_correct_testing_set=100-percentage_wrong_testing_set;
%disp('')
%disp(['Percentage  right for  KNN ',num2str(percentage_correct_testing_set),' %'])
mm(k)=percentage_correct_testing_set;


end


% Determine the best performance for KNN Test data
[m1,idx] = max(mm, [], 2);

% Since I already determined my best k, I'm going to classify using it and
% form my risk Group 1 using it.
clear k_cluster_classification_testing_set
k_cluster_classification_testing_set = knnclassify(full_sample_testing_set, training, group,idx) ;

malign_patient_amount_knn_classification=size(k_cluster_classification_testing_set(k_cluster_classification_testing_set==1),1); % This is how many patiente classify as malignant


pmd=100*MDdiagnostic/size(fully_sanitized_data,1);
performance_knn_classification_testing_set=100*malign_patient_amount_knn_classification/size(fully_sanitized_data,1);
comp=pmd-performance_knn_classification_testing_set;

 % Display results KNN Test Set


 disp('   ')
 disp(['The best performance for KNN Classification'])
 disp(['using for the Test Set  was  ',num2str(100-comp),' % right with ',num2str(idx), ' iteration(s)'])
 disp(['out of ', num2str(knn_iterations),' iterations'])
% disp('    KNN for the Test set ( 60% )   ')
 %disp(['There are   ',num2str(malign_patient_amount_knn_classification),' patients'])
 %disp('with probable malignant diagnostic ')
 %disp('according to the KNN for the Test Set')



 disp('    ')
 %disp(['The KNN classifier diagnostic for the test set differs  ',num2str(comp),' % '])
 %disp('of patients with probable malignant diagnostic from ')
 %disp('the Doctors diagnostic')
 %disp('----------------------------------')


 % set up the domain over which you want to visualize the decision
% boundary
clear ns
ns = size(randomized_testing_set(:,3),1);


xrange = [1 ns];
yrange = [1 5];

incr = 1;

pat60=1:ns;

% create some training data for three classes.
class_testing_set = cell(3,1);
class_testing_set{1} =[randomized_testing_set(:,3) pat60'];
class_testing_set{2} =[randomized_testing_set(:,4)  pat60'];
class_testing_set{3} = [randomized_testing_set(:,5) pat60'] ;


%% show the image KNN for Test Set

figure('Name','KNN for Test Set','NumberTitle','off','Position',[1 scrsz(4)  scrsz(3) scrsz(4)])
hold on;
set(gca,'ydir','normal');

 %plot the class test data.
 plot(class_testing_set{1}(:,2),class_testing_set{1}(:,1), 'c.','MarkerSize',10);
 plot(class_testing_set{2}(:,2),class_testing_set{2}(:,1), 'g.','MarkerSize',10);
 plot(class_testing_set{3}(:,2),class_testing_set{3}(:,1), 'm.','MarkerSize',10);

% plot the classification
plot(pat60(k_cluster_classification_testing_set==0),full_sample_testing_set(k_cluster_classification_testing_set==0),'ob','LineWidth',.2,'MarkerSize',20);
plot(pat60(k_cluster_classification_testing_set==1),full_sample_testing_set(k_cluster_classification_testing_set==1),'or','LineWidth',.2,'MarkerSize',20);


xlabel(['Patient by number.  ' num2str( max(ns) ) ' patients displayed'] ,'FontSize',25)
ylabel('  Parameters Shape/Margin/Density ','FontSize',25)
title('KNN Classifier for the Test Set','FontSize',25)


le = legend('Shape','Margin','Density','Patients Classified as Benign',...
 'Patients Classified as Malign',2,'location','North', 'orientation', 'horizontal');
set(le,'Interpreter','none')
title({'';[...
  'KNN Classifier for the Test Set classified  successfully  ',num2str(100-comp)...
  ' % of the patients ']},'fontsize',16)

% End of KNN



%% Linear discriminant analysis (LDA). Training Set


%{
class = classify(sample,training,group,'type') allows you to specify the type of discriminant function. Specify type inside single quotes. type is one of:

linear — Fits a multivariate normal density to each group, with a pooled estimate of covariance. This is the default.

quadratic — Fits multivariate normal densities with covariance estimates stratified by group.
Using Mahalanobis distances computed from the groupspecific
covariance matrices leads to an allocation rule that
draws quadratic boundaries between groups in the variable
space, so this is called quadratic discriminant analysis. It
can be implemented using the classify function with type
set to ‘quadratic’. The Matlab documentation refers to the
individual-group covariance estimates as “stratified by
group”.
%}

 patient_amount_training_set_lda = size(randomized_training_set,1); % This is the amount of patients on the training set
 indexes_patient_amount_training_set_lda=[1:patient_amount_training_set_lda]; % This is the index for patients on the training set

 measu40=[ randomized_training_set(:,3)   randomized_training_set(:,4)  randomized_training_set(:,5)];

 % Classification
 [ldaClass40] = classify(measu40,training,group);

 [ldaClassQ40] = classify(measu40,training,group,'quadratic');


  % Determining the size of the Group classified as malign using LDA
  % training set
  v=0;v1=0;
  for m=1:size(ldaClass40,1)
    if ldaClass40(m)==1
     v=v+1;   malignclass40(v)=ldaClass40(m);
   end;
   if ldaClassQ40(m)==1
     v1=v1+1;   malignclassQ40(v1)=ldaClassQ40(m);
   end;




 end;

    % Creating the risk MDdiagnostic classify as malignant by Doctors severity
  % attribute for training set in order to compare results


  h=0; Group7=zeros(10,6);
  for j=1:length(randomized_training_set);

    if randomized_training_set(j,6)==1
     h=h+1;
     Group7(h,:)= randomized_training_set(j,:);
   end;

 end

 MDdiagnostic40=size(Group7,1);

 compclass40=abs((size(malignclass40,2)-length(Group7))*100/(size(malignclass40,2)));
 compclassQ40=abs((size(malignclassQ40,2)-length(Group7))*100/(size(malignclassQ40,2)));


%% LDA Training Set Results

disp('    ')
disp('-----------------   ooo   --------------')
disp('    ')

% disp(['The Linear discriminant analysis for the Training Set differs  ',num2str(compclass40),' % '])
 %disp('of patients with probable malignant diagnostic ')
 % disp('from the Doctors diagnostic ')

 disp(['Linear discriminant analysis classified  successfully  ',num2str( 100-compclass40 ),' %'])
 disp('  of the patients for the Training Set ')

 disp('    ')
  %disp(['The Quadratic discriminant analysis diagnostic for the Training Set differs  ',num2str(compclassQ40),' % '])
 %disp('of patients with probable malignant diagnostic ')
 % disp('from the Doctors diagnostic ')
 disp(['Linear discriminant quadratic classified  successfully  ',num2str( 100-compclassQ40 ),' %'])
 disp(' of the patients for the Training Set ')
 disp('   ')
 bad40 = ~strcmp(ldaClass40,randomized_training_set(:,3));
 ldaResubErr = sum(bad40) / patient_amount_training_set_lda;

% End of LDA Training Set

  %% Linear discriminant analysis (LDA). Test Set


 N60 = size(randomized_testing_set,1); % This is the amount of patients on the test set
 pat60=[1:N60]; % This is index for patients on the test set

 measu60=[ randomized_testing_set(:,3)   randomized_testing_set(:,4)  randomized_testing_set(:,5)];
 g=measu60;

  % Classification of the test set
  [ldaClass60] = classify(measu60,training,group);

  [ldaClassQ60] = classify(measu60,training,group,'quadratic');

  % Determining the size of the Group classified as malign using LDA
  % training set
  v=0;v1=0;
  for m=1:size(ldaClass60,1)
    if ldaClass60(m)==1
     v=v+1;   malignclass60(v)=ldaClass60(m);
   end;
   if ldaClassQ60(m)==1
     v1=v1+1;   malignclassQ60(v1)=ldaClassQ60(m);
   end;


 end;


    % Creating the risk MDdiagnostic classify as malignant by Doctors severity
  % attribute for testing set in order to compare results


  h=0; Group8=zeros(10,6);
  for j=1:length(randomized_testing_set);

    if randomized_testing_set(j,6)==1
     h=h+1;
     Group8(h,:)= randomized_testing_set(j,:);
   end;

 end


 MDdiagnostic60=size(Group8,1);


 compclass60=abs((size(malignclass60,2)-length(Group8))*100/(size(malignclass60,2)));
 compclassQ60=abs((size(malignclassQ60,2)-length(Group8))*100/(size(malignclassQ60,2)));


%% LDA Test Set  Results

disp('    ')
disp('-----------------   ooo   --------------')
disp('    ')

 %disp(['The Linear discriminant analysis for the Test Set differs  ',num2str(compclass60),' % '])
% disp('of patients with probable malignant diagnostic ')
 % disp('from the Doctors diagnostic ')

 disp(['Linear discriminant analysis classified  successfully  ',num2str( 100-compclass60 ),' %'])
 disp(' of the patients for the Test Set ')

 disp('    ')
 % disp(['The Quadratic discriminant analysis diagnostic for the Test Set differs  ',num2str(compclassQ60),' % '])
 %disp('of patients with probable malignant diagnostic ')
 % disp('from the Doctors diagnostic ')
 disp(['Linear discriminant quadratic classified  successfully  ',num2str( 100-compclassQ60 ),' %'])
 disp('  of the patients for the Test Set ')

 bad60 = ~strcmp(ldaClass60,randomized_testing_set(:,3));
 ldaResubErr60 = sum(bad60) / N60;

% End of LDA Test Set


   %% Display raw data graph
   scrsz = get(0,'ScreenSize');

  % Raw  Patient  data
  figure('Name','Sample of Raw  Patient Data','NumberTitle','off','Position',[1 scrsz(4)  scrsz(3) scrsz(4)])
  hold all

 sampleg=1:30; % Amount of raw data displayed, smaller sample for clarity


 plot(indexes_patient_amount_training_set_lda(sampleg),measu40(sampleg,1),'ob','LineWidth',2,'MarkerSize',20); % Plot Shape
   plot(indexes_patient_amount_training_set_lda(sampleg),measu40(sampleg,2),'diamondg','LineWidth',2,'MarkerSize',20);  % Plot Margin
      plot(indexes_patient_amount_training_set_lda(sampleg),measu40(sampleg,3),'xr','LineWidth',2,'MarkerSize',20);  % Plot Margin

 % Formatting Axis and Legends

 le = legend('parameters Shape','Parameters Margin','Parameters Density',2,'location','North', 'orientation', 'horizontal');
 set(le,'Interpreter','none')



 xlabel(['Patient by number.  ' num2str( max(sampleg) ) ' patients displayed'] ,'FontSize',25)
 ylabel('  Parameters Shape/Margin/Density ','FontSize',25)
 title('Raw Data','FontSize',25)
 xs=size(sampleg,2);
 axis([ 0 xs 0 6])


  % End of LDA


%% K means Clustering Training Set

% Initialization. % Since I am trying to classify between Benign40 or malign
% conditions, my Codebook will be the average for malign and the average for
% Benign for every class assuming equal priors.


% Average for Shape Benign40 & malign
muSB40=mean(randomized_training_set(randomized_training_set(:,6)==0 ,3));
muSM40=mean(randomized_training_set(randomized_training_set(:,6)==1 ,3));

% Average for Margin Benign40 & malign
muMB40=mean(randomized_training_set(randomized_training_set(:,6)==0 ,4));
muMM40=mean(randomized_training_set(randomized_training_set(:,6)==1 ,4));

% Average for Density Benign40 & malign
muDB40=mean(randomized_training_set(randomized_training_set(:,6)==0 ,5));
muDM40=mean(randomized_training_set(randomized_training_set(:,6)==1 ,5));


  % Initial distortion distances_to_training_set_means stores all the distances to the corresponding
  % means
  distances_to_training_set_means=[ dist(randomized_training_set(:,3),muSB40),...
  dist(randomized_training_set(:,4),muSB40),...
  dist(randomized_training_set(:,5),muSB40),...
  dist(randomized_training_set(:,3),muSM40),...
  dist(randomized_training_set(:,4),muSM40),...
  dist(randomized_training_set(:,5),muSM40)] ;

  Distortinitial40=sum(distances_to_training_set_means);
Distortinitial40=sum(Distortinitial40); % Storing distortion

[ dst40,class40] =min(distances_to_training_set_means,[],2);

 % All benign patients classified from the training  set
 Benign40 =   cat( 1, randomized_training_set(class40==1,: ),...
   randomized_training_set(class40==2,: ) ,...
   randomized_training_set(class40==3,: ) );

 % All malign patients classified from the training  set

 training_set_malign_patients =   cat( 1, randomized_training_set(class40==4,: ),...
   randomized_training_set(class40==5,: ) ,...
   randomized_training_set(class40==6,: ) );


   %  Codebook update
   tdistortion40=zeros(2);
   for iter40=1:size(randomized_training_set,1)

% Average for Shape Benign & malign
muSB40=mean(Benign40(: ,3));
muSM40=mean(training_set_malign_patients (:,3));

% Average for Margin Benign & malign
muMB40=mean(Benign40(: ,4));
muMM40=mean(training_set_malign_patients (:,4));

% Average for Density Benign & malign
muDB40=mean(Benign40(: ,5));
muDM40=mean(training_set_malign_patients (:,5));


  %% Termination

  distances_to_training_set_means=[ dist(randomized_training_set(:,3),muSB40),...
  dist(randomized_training_set(:,4),muSB40),...
  dist(randomized_training_set(:,5),muSB40),...
  dist(randomized_training_set(:,3),muSM40),...
  dist(randomized_training_set(:,4),muSM40),...
  dist(randomized_training_set(:,5),muSM40)] ;

  distortion=sum(distances_to_training_set_means);
  tdistortion40(iter40)=sum(distortion);

  if iter40>=2 && tdistortion40(iter40-1)==tdistortion40(iter40)
    break
  end


end;

% Display results for training set

disp('    ')
disp('-----------------   ooo   --------------')
disp('    ')


display ('K means Clustering analysis for Training set')
display (['The distortion converges after ',num2str(iter40),' iterations'])

disp('   ')

 % diff = ~strcmp(ldaClass40(ldaClass40==1),randomized_training_set(randomized_training_set==1,6));

 compclassKmeans40=abs(length(Group7)-size(training_set_malign_patients,1))*100/(size(training_set_malign_patients,1));

 %disp(['The K means Clustering analysis diagnostic differs  ',num2str(compclassKmeans40),' % '])
% disp('of patients with probable malignant diagnostic ')
  %disp('from the Doctors diagnostic ')
  disp(['The K means Clustering analysis  successfully  classified  ',num2str(100-compclassKmeans40),' % '])

  %disp(['The best performing classification was  ',num2str(compclass),' % '])



  %% Show the image K means Training Set
  scrsz = get(0,'ScreenSize');

  figure('Name','K means Classifier for the Training Set','NumberTitle','off','Position',[1 scrsz(4)  scrsz(3) scrsz(4)])
  hold all

  hold on;
  set(gca,'ydir','normal');
 %plot the class training data.
 plot(train40{1}(:,2),train40{1}(:,1), 'c.','MarkerSize',10);
 plot(train40{2}(:,2),train40{2}(:,1), 'g.','MarkerSize',10);
 plot(train40{3}(:,2),train40{3}(:,1), 'm.','MarkerSize',10);



 for i=1:length(dst40)
   hold all
   if dst40(i)>=0.3
    plot(indexes_patient_amount_training_set_lda(i),randomized_training_set(i,3:5),'ob','LineWidth',.1,'MarkerSize',20)
  end

  if dst40(i)<0.3
   plot(indexes_patient_amount_training_set_lda(i),randomized_training_set(i,3:5),'or','LineWidth',.1,'MarkerSize',20)
 end

end;


  % Formatting Axis and Legends

  xlabel(['Patient by number.  ' num2str( max(indexes_patient_amount_training_set_lda) ) ' patients displayed'] ,'FontSize',25)
  ylabel('  Parameters Shape/Margin/Density ','FontSize',25)
  title('Kmeans Classifier for the Training Set','FontSize',25)
  title({'';[...
    ' Kmeans Classifier for the Training Set classified successfully ', num2str(100-compclassKmeans40)...
    ' % of the patients ']},'fontsize',16)

  le = legend('Shape','Margin','Density','Patients Classified as Benign',...
   'Patients Classified as Malign(Red) ','location','North', 'orientation', 'horizontal');
  set(le,'Interpreter','none')

 % End of Kmeans Training set



%% K means Clustering Test set

% Initialization. % Since I am trying to classify between Benign or malign
% conditions, my Codebook will be the average for malign and the average for
% Benign60 for every class.

% Average for Shape Benign60 & malign
muSB60=mean(randomized_testing_set(randomized_testing_set(:,6)==0 ,3));
muSM60=mean(randomized_testing_set(randomized_testing_set(:,6)==1 ,3));

% Average for Margin Benign60 & malign
muMB60=mean(randomized_testing_set(randomized_testing_set(:,6)==0 ,4));
muMM60=mean(randomized_testing_set(randomized_testing_set(:,6)==1 ,4));

% Average for Density Benign60 & malign
muDB60=mean(randomized_testing_set(randomized_testing_set(:,6)==0 ,5));
muDM60=mean(randomized_testing_set(randomized_testing_set(:,6)==1 ,5));


  % Initial distortion
  d60=[ dist(randomized_testing_set(:,3),muSB60),...
  dist(randomized_testing_set(:,4),muSB60),...
  dist(randomized_testing_set(:,5),muSB60),...
  dist(randomized_testing_set(:,3),muSM60),...
  dist(randomized_testing_set(:,4),muSM60),...
  dist(randomized_testing_set(:,5),muSM60)] ;

  Distortinitial60=sum(d60);
  Distortinitial60=sum(Distortinitial60);

  [ dst60,class60] =min(d60,[],2);

 % All benign patients classified from the test set

 Benign60 =   cat( 1, randomized_testing_set(class60==1,: ),...
   randomized_testing_set(class60==2,: ) ,...
   randomized_testing_set(class60==3,: ) );

  % All malign patients classified from the test set
  Malign60 =   cat( 1, randomized_testing_set(class60==4,: ),...
   randomized_testing_set(class60==5,: ) ,...
   randomized_testing_set(class60==6,: ) );



   %  Codebook update
   tdistortion60=zeros(2);
   for iter60=1:size(randomized_testing_set,1)

% Average for Shape Benign & malign
muSB60=mean(Benign60(: ,3));
muSM60=mean(Malign60 (:,3));

% Average for Margin Benign & malign
muMB60=mean(Benign60(: ,4));
muMM60=mean(Malign60 (:,4));

% Average for Density Benign & malign
muDB60=mean(Benign60(: ,5));
muDM60=mean(Malign60 (:,5));


  %% Termination

  d60=[ dist(randomized_testing_set(:,3),muSB60),...
  dist(randomized_testing_set(:,4),muSB60),...
  dist(randomized_testing_set(:,5),muSB60),...
  dist(randomized_testing_set(:,3),muSM60),...
  dist(randomized_testing_set(:,4),muSM60),...
  dist(randomized_testing_set(:,5),muSM60)] ;

  distortion60=sum(d60);
  tdistortion60(iter60)=sum(distortion);

  if iter60>=2 && tdistortion60(iter60-1)==tdistortion60(iter60)
    break
  end


end;

% Display results for Test set


disp('    ')
disp('-----------------   ooo   --------------')

disp('   ')

display ('K means Clustering analysis for Test set')
display (['The distortion converges after ',num2str(iter60),' iterations'])


disp('   ')

 % diff = ~strcmp(ldaClass60(ldaClass60==1),randomized_testing_set(randomized_testing_set==1,6));

 unsuccessful_kmeans_test_set=abs(length(Group8)-size(Malign60,1))*100/(size(Malign60,1));



% disp(['The K means Clustering analysis diagnostic differs  ',num2str(unsuccessful_kmeans_test_set),' % '])
 %disp('of patients with probable malignant diagnostic ')
 % disp('from the Doctors diagnostic ')
 disp(['The K means Clustering analysis  successfully  classified  ',num2str(100-unsuccessful_kmeans_test_set),' % '])

  %disp(['The best performing classification was  ',num2str(compclass),' % '])
  disp('   ')



  %% Conclusion and Analysis

  disp('-------------- Conclusions ---------------')
  disp('   ')
  perf=[  100-comp   100-compclass60   100-unsuccessful_kmeans_test_set ];

  [Klass Best]  =max(perf,[],2);

  if Best==1
   disp(['The best classifier was KNN with ',num2str(Klass),' % of patients'])
   disp('successfully classified ')

 else if Best==2

   disp(['The best classifier was LDA with ',num2str(Klass),' % of patients'])
   disp('              successfully classified. ')

 else

  disp(['The best classifier was K means with ',num2str(Klass),' % of patients'])
  disp('successfully classified ')

end
end
disp('   ')
disp('      -----------  END  -------------')



  %% Show the image K means Test Set
  scrsz = get(0,'ScreenSize');
  figure('Name','K means Classifier for the Test Set','NumberTitle','off','Position',[1 scrsz(4)  scrsz(3) scrsz(4)])
  hold all

  hold on;
%set(gca,'ydir','normal');



 %plot the class training data.
 plot(class_testing_set{1}(:,2),class_testing_set{1}(:,1), 'c.','MarkerSize',10);
 plot(class_testing_set{2}(:,2),class_testing_set{2}(:,1), 'g.','MarkerSize',10);
 plot(class_testing_set{3}(:,2),class_testing_set{3}(:,1), 'm.','MarkerSize',10);


 for i=1:length(dst60)
   hold all
   if dst60(i)>=0.3
    plot(pat60(i),randomized_testing_set(i,3:5),'ob','LineWidth',.1,'MarkerSize',15)
  end

  if dst60(i)<0.3
   plot(pat60(i),randomized_testing_set(i,3:5),'or','LineWidth',.1,'MarkerSize',15)
 end

end;

  % Formatting Axis and Legends

  xlabel(['Patient by number.  ' num2str(max(pat60)) ' patients displayed'] ,'FontSize',25)
  ylabel('  Parameters Shape/Margin/Density ','FontSize',25)
  title('Kmeans Classifier for the Test Set','FontSize',25)
  title({'';[...
    ' Kmeans Classifier for the Test Set classified successfully ', num2str(100-unsuccessful_kmeans_test_set)...
    ' % of the patients ']},'fontsize',16)

  le = legend('Shape','Margin','Density','Patients Classified as Benign',...
   'Patients Classified as Malign(Red) ','location','North', 'orientation', 'horizontal');
  set(le,'Interpreter','none')

 % End of Kmeans Test set


%%   LDA Display Graph Training Set

% number of training samples
training_samples_number = size(randomized_training_set,1);
patient_amount_training_set_lda = size(randomized_training_set,1);
indexes_patient_amount_training_set_lda=[1:patient_amount_training_set_lda];



%  Assign training data for three classes.
class_training_data_set_lda = cell(3,1);
class_training_data_set_lda{1} =[randomized_training_set(:,3) indexes_patient_amount_training_set_lda'];
class_training_data_set_lda{2} =[randomized_training_set(:,4)  indexes_patient_amount_training_set_lda'];
class_training_data_set_lda{3} = [randomized_training_set(:,5) indexes_patient_amount_training_set_lda'] ;

% sample mean
sample_means_training_set = cell(length(class_training_data_set_lda),1);

% compute sample mean to use as the class prototype.
for i=1:length(class_training_data_set_lda),
  sample_means_training_set{i} = mean(class_training_data_set_lda{i});
end

% set up the domain over which you want to visualize the decision
% boundary
xrange = [1 5];
yrange = [1 training_samples_number];
% step size for how finely you want to visualize the decision boundary.
inc = 1;

% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);

xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.



xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];


numxypairs = length(xy); % number of (x,y) pairs

% distance measure evaluations for each (x,y) pair.
dist = [];

% loop through each class and calculate distance measure for each (x,y)
% from the class prototype.
for i=1:length(class_training_data_set_lda),

    % calculate the city block distance between every (x,y) pair and
    % the sample mean of the class.
    % the sum is over the columns to produce a distance for each (x,y)
    % pair.
    disttemp = sum(abs(xy - repmat(sample_means_training_set{i}, [numxypairs 1])), 2);

    % concatenate the calculated distances.
    dist = [dist disttemp];

  end

% for each (x,y) pair, find the class that has the smallest distance.
% this will be the min along the 2nd dimension.
[m,idx] = min(dist, [], 2);

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(idx, image_size);



% show the image Linear discriminant analysis Training Set

figure('Name','Linear discriminant analysis Training Set','NumberTitle','off','Position',[1 scrsz(4)  scrsz(3) scrsz(4)])


imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');

% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue

colormap([1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1]);

% plot the class training data.
plot(class_training_data_set_lda{1}(:,1),class_training_data_set_lda{1}(:,2), 'r.');
plot(class_training_data_set_lda{2}(:,1),class_training_data_set_lda{2}(:,2), 'go');
plot(class_training_data_set_lda{3}(:,1),class_training_data_set_lda{3}(:,2), 'b*');

legend(' Shape ', 'Margin', 'Density','Location','North', ...
  'Orientation', 'horizontal');
title({'Linear discriminant analysis Training Set';[...
  'Linear discriminant analysis classified  successfully  ',num2str( 100-compclass40 )...
  ' % of the patients ']},'fontsize',16)

 % Formatting Axis and Legends

 xlabel('Parameters','fontsize',25);

 ylabel('Patient Number','fontsize',25);

 x = [.25 .44];
 y = [0.3 0.5];
% Create the textarrow object:
txtar =  annotation('textarrow',x,y,'String','Decision Line','FontSize',14);

text(1.2,170,'Malignant Cases','FontSize',14,'color','r')
text(3.2,170,'Benign Cases','FontSize',14,'color','b')





%% LDA Display Graph Test Set
 %

% number of test samples

test_samples_number = size(randomized_testing_set,1);
N60 = size(randomized_testing_set,1);
pat60=[1:N60];



%  Assign test data for three classes.
class_test_data_set_lda = cell(3,1);
class_test_data_set_lda{1} =[randomized_testing_set(:,3) pat60'];
class_test_data_set_lda{2} =[randomized_testing_set(:,4)  pat60'];
class_test_data_set_lda{3} = [randomized_testing_set(:,5) pat60'] ;

% sample mean
sample_means = cell(length(class_test_data_set_lda),1);

% compute sample mean to use as the class prototype.
for i=1:length(class_test_data_set_lda),
  sample_means{i} = mean(class_test_data_set_lda{i});
end

% set up the domain over which you want to visualize the decision
% boundary
xrange = [1 5];
yrange = [1 test_samples_number];
% step size for how finely you want to visualize the decision boundary.
inc = 1;

% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);

xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.



xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];


numxypairs = length(xy); % number of (x,y) pairs

% distance measure evaluations for each (x,y) pair.
dist = [];

% loop through each class and calculate distance measure for each (x,y)
% from the class prototype.
for i=1:length(class_test_data_set_lda),

    % calculate the city block distance between every (x,y) pair and
    % the sample mean of the class.
    % the sum is over the columns to produce a distance for each (x,y)
    % pair.
    disttemp = sum(abs(xy - repmat(sample_means{i}, [numxypairs 1])), 2);

    % concatenate the calculated distances.
    dist = [dist disttemp];

  end

% for each (x,y) pair, find the class that has the smallest distance.
% this will be the min along the 2nd dimension.
[m,idx] = min(dist, [], 2);

% reshape the idx (which contains the class label) into an image.
decisionmap60 = reshape(idx, image_size);



%show the image
figure('Name','Linear discriminant analysis Test Set',...
 'NumberTitle','off','Position',[1 scrsz(4)  scrsz(3) scrsz(4)])


imagesc(xrange,yrange,decisionmap60);
hold on;
set(gca,'ydir','normal');

% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue

colormap([1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1]);

% plot the class training data.
plot(class_test_data_set_lda{1}(:,1),class_test_data_set_lda{1}(:,2), 'r.');
plot(class_test_data_set_lda{2}(:,1),class_test_data_set_lda{2}(:,2), 'go');
plot(class_test_data_set_lda{3}(:,1),class_test_data_set_lda{3}(:,2), 'b*');



legend(' Shape ', 'Margin', 'Density','Location','North', ...
  'Orientation', 'horizontal');
title({'Linear discriminant analysis Test Set';[...
  'Linear discriminant analysis classified  successfully  ',num2str( 100- compclass60 )...
  ' % of the patients ']},'fontsize',16)

 % Formatting Axis and Legends

 xlabel('Parameters','fontsize',25);

 ylabel('Patient Number','fontsize',25);

 x = [.25 .44];
 y = [0.3 0.5];
% Create the textarrow object:
txtar =  annotation('textarrow',x,y,'String','Decision Line','FontSize',14);

text(1.2,170,'Malignant Cases','FontSize',14,'color','r')
text(3.2,170,'Benign Cases','FontSize',14,'color','b')

% End of code