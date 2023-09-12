#!/bin/bash
# sessname="temporal_joint_trainings"
sessname=$1

# cd ~/data
# mkdir "$1"
cd ../

# Create a new session named "$sessname", and run command
# tmux new-session -d -s "$sessname" || echo "going to the session $sessname"
tmux new-session -d -s "bet1" || echo "going to the session $sessname"
tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=1000 object=bowl_picking device=0 vision_view_num=1" Enter

tmux new-session -d -s "bet2" || echo "going to the session $sessname"
tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=1000 object=plier_picking device=1 vision_view_num=0" Enter

tmux new-session -d -s "bet3" || echo "going to the session $sessname"
tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=1000 object=peg_insertion device=2 vision_view_num=0" Enter

tmux new-session -d -s "bet4" || echo "going to the session $sessname"
tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=1000 object=card_turning device=3 vision_view_num=0" Enter
tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=1000 object=card_flipping device=3 vision_view_num=0" Enter



# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=$4 device=$2 vision_view_num=$5" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=card_flipping device=$2" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=peg_insertion device=$2" Enter


# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=bowl_picking learner.total_loss_type=$2 device=$3" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=plier_picking learner.total_loss_type=$2 device=$3" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=card_flipping learner.total_loss_type=$2 device=$3" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=peg_insertion learner.total_loss_type=$2 device=$3" Enter

# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=card_turning device=$2" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=plier_picking device=$2" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=card_flipping device=$2" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=peg_insertion device=$2" Enter

# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=bowl_picking learner.total_loss_type=contrastive device=$3 vision_view_num=1" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=300 object=bowl_picking learner.total_loss_type=joint device=$3 vision_view_num=1" Enter

# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=card_turning learner.total_loss_type=contrastive device=$2" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=peg_insertion learner.total_loss_type=joint device=$2" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=peg_insertion learner.total_loss_type=contrastive device=$2" Enter

# TODO: Bowl picking gave error since you didn't change the view - fix that

# Attach to session named "$sessname"
#tmux attach -t "$sessname"