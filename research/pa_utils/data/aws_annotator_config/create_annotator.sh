#!/bin/bash

# sudo userdel -f annotator3
# sudo rm -rf /home/annotator3
# sudo rm /tmp/.X*-lock
# sudo rm /tmp/.X11-unix/X*

ANNOTATOR_NUM=$1
USERNAME=annotator$ANNOTATOR_NUM;

#sudo userdel $USERNAME
#sudo rm -rf /home/$USERNAME

userCommand()
{
    echo sudo su - $USERNAME -c "$2"
    sudo su - $USERNAME -c "$2"
}


# create annotator
if [ -z $USERNAME ]
then
	echo Please specify username
	return
fi
echo creating user: $USERNAME

sudo useradd -m $USERNAME
sudo passwd $USERNAME

if [ ! -d "$/HOME/Desktop" ]
then
	echo create Desktop directory for $HOME
    userCommand $USERNAME 'mkdir $HOME/Desktop'
fi
echo copy /opt/* to $HOME/Desktop
userCommand $USERNAME 'cp /opt/Tools/labelImg.sh $HOME/Desktop'
userCommand $USERNAME 'chmod +x $HOME/Desktop/labelImg.sh'
userCommand $USERNAME 'cp /opt/Tools/labelImg/data/predefined_classes.txt $HOME/Desktop'
userCommand $USERNAME 'cp -r /opt/LabelResult $HOME/Desktop'


# make sure annotator cannot access other's directory
sudo chmod o-x /home/*

echo Create VNC for $USERNAME..
userCommand $USERNAME 'vncserver :'"$ANNOTATOR_NUM"
userCommand $USERNAME 'vncserver -kill :'"$ANNOTATOR_NUM"

userCommand $USERNAME 'mv ~/.vnc/xstartup ~/.vnc/xstartup.bak'
userCommand $USERNAME 'echo "#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &" >> ~/.vnc/xstartup'

userCommand $USERNAME 'chmod +x $HOME/.vnc/xstartup'

sudo systemctl enable vncserver@"$ANNOTATOR_NUM".service
sudo systemctl start vncserver@"$ANNOTATOR_NUM"