import numpy as np
import matplotlib.pyplot as plt
from hsdl.hsdl import plot_confusionmatrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test=np.load('RF1_y_test.npy')
y_pred=np.load('RF1_y_pred.npy')
cm = confusion_matrix(y_test, y_pred)

f, axes = plt.subplots(1, 2, figsize=(20, 10))#, sharey='row')
axes[0].text(-0.15,1,'(a)',transform=axes[0].transAxes,size=20,weight='normal')
axes[1].text(-0.15,1,'(b)',transform=axes[1].transAxes,size=20,weight='normal')

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp1.plot(ax=axes[0], xticks_rotation=45,colorbar=False, text_kw={'size': 16, 'weight': 'bold', 'color': 'darkred'})
disp1.ax_.set_title('Random Forest Classifier',size=20,weight='normal')
disp1.ax_.set_xlabel('Predicted label',size=20,weight='normal')
disp1.ax_.set_ylabel('',size=20,weight='normal')
disp1.ax_.xaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');
disp1.ax_.yaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');
# disp1.ax_.
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test=np.load('XGBoost1_y_test.npy')
y_pred=np.load('XGBoost1_y_pred.npy')
cm = confusion_matrix(y_test, y_pred)

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp2.plot(ax=axes[1], xticks_rotation=45,colorbar=False,text_kw={'size': 16, 'weight': 'bold', 'color': 'darkred'})
disp2.ax_.set_title('XGBoost Classifier',size=20,weight='normal')
disp2.ax_.set_xlabel('Predicted label',size=20,weight='normal')
disp2.ax_.set_ylabel('',size=20,weight='normal')
disp2.ax_.xaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');
disp2.ax_.yaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');

f.add_axes([0.23,0.9,0.3,0.2]);f.gca().axis('off')
plt.text(0.0,0.0,'Confusion matrix comparison for training database 1',fontsize=30,weight='normal')

f.add_axes([0.1,0.02,0.3,0.2]);f.gca().axis('off')
plt.text(0.1,0.05,'Fake: 200; Real: 200; Total: 400 (280 for training and 120 for testing)',fontsize=12,color='k')
plt.savefig('XGBoost-cm-data1.png')
plt.show()


# import numpy as np
# cf_matrix=np.array([[13596,   947],
#        [  296,  8141]])
#after:
# cf_matrix=np.array([[14373,   170],
#  [  158,  8279]])
# plot_confusionmatrix(cf=cm,categories=['Fake','Real'],figname='XGBoost-cm-data2.png',ifshow=False)



# plot_confusionmatrix(cf=None,categories=['Up','Down'],figname='Conf_Matrix_before_transferlearning.png',ifshow=True):




import numpy as np
import matplotlib.pyplot as plt
from aaspip import plot_confusionmatrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test=np.load('RF1_y_test_8features.npy')
y_pred=np.load('RF1_y_pred_8features.npy')
cm = confusion_matrix(y_test, y_pred)

f, axes = plt.subplots(1, 2, figsize=(20, 10))#, sharey='row')
axes[0].text(-0.15,1,'(a)',transform=axes[0].transAxes,size=20,weight='normal')
axes[1].text(-0.15,1,'(b)',transform=axes[1].transAxes,size=20,weight='normal')

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp1.plot(ax=axes[0], xticks_rotation=45,colorbar=False, text_kw={'size': 16, 'weight': 'bold', 'color': 'darkred'})
disp1.ax_.set_title('Random Forest Classifier',size=20,weight='normal')
disp1.ax_.set_xlabel('Predicted label',size=20,weight='normal')
disp1.ax_.set_ylabel('',size=20,weight='normal')
disp1.ax_.xaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');
disp1.ax_.yaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');
# disp1.ax_.
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test=np.load('XGBoost1_y_test_8features.npy')
y_pred=np.load('XGBoost1_y_pred_8features.npy')
cm = confusion_matrix(y_test, y_pred)

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp2.plot(ax=axes[1], xticks_rotation=45,colorbar=False,text_kw={'size': 16, 'weight': 'bold', 'color': 'darkred'})
disp2.ax_.set_title('XGBoost Classifier',size=20,weight='normal')
disp2.ax_.set_xlabel('Predicted label',size=20,weight='normal')
disp2.ax_.set_ylabel('',size=20,weight='normal')
disp2.ax_.xaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');
disp2.ax_.yaxis.set_ticklabels(['Fake', 'Real'],size=14,weight='normal');

f.add_axes([0.23,0.9,0.3,0.2]);f.gca().axis('off')
plt.text(0.0,0.0,'Confusion matrix comparison for training database 1',fontsize=30,weight='normal')

f.add_axes([0.1,0.02,0.3,0.2]);f.gca().axis('off')
plt.text(0.1,0.05,'Fake: 200; Real: 200; Total: 400 (280 for training and 120 for testing); 8 Features are used in classification',fontsize=12,color='k')
plt.savefig('XGBoost-cm-data1-8features.png')
plt.show()


