import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView } from 'react-native';
import { useDispatch } from 'react-redux';
import {deleteCardAction} from '../redux/actions'

const InformationCard = (props) => {
const dispatch = useDispatch();
const deleteCard = () => {
    dispatch(deleteCardAction(props.index))
}
        return(
            <TouchableOpacity
            onPress={props.navigation}>
                <SafeAreaView>
                    <View style = {styles.infoContainer}>
                        <Text style = {styles.infoContext}>
                            {props.name}
                        </Text>
                        <TouchableOpacity onPress={deleteCard}>
                            <Text style = {styles.infoContext}>
                                Del
                             </Text>
                        </TouchableOpacity> 
                    </View>
                </SafeAreaView>
            </TouchableOpacity>                 
        );

}

const styles = StyleSheet.create({
    infoContainer:{
        paddingTop: 8,
        marginBottom: 20,
        height: 40,
        width: 325,
        backgroundColor: '#80d8ff',
        alignSelf: "center",
        borderRadius: 10,
        flexWrap: 'wrap',
        flexDirection:'row',
        justifyContent: 'space-between', 
    },
    infoContext:{
        fontSize: 15,
        fontWeight:"900",
        color: '#FFF',
        marginHorizontal: 20,
    },
});

export default InformationCard;