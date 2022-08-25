import React from 'react';
import {Text, StyleSheet, SafeAreaView, View, FlatList } from 'react-native';
import InformationCard from '../components/InformationCard';
import AddButton from '../components/AddButton';
import { useSelector } from 'react-redux';

const Home = ({ navigation }) => {

    const informationList = useSelector((state) => state.people)
    return (
        <SafeAreaView style = {styles.container}>
            <View style = {styles.titleWrapper}>
                <Text style = {styles.title}>
                    Information
                </Text>
            </View>
           
            <FlatList 
            extraData={informationList}
            data = {informationList}
            renderItem = {({item, index}) => {
                return(
                    <InformationCard 
                        name = {item.name} 
                        navigation = {() => navigation.navigate("Modify", {isAdd: false, i: index})}
                        index = {index}
                    />
                )
            }} />

            <AddButton 
            navigation = {() => navigation.navigate("Modify", {isAdd: true})}
            />
        </SafeAreaView>
    );
}
const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#e3f2fd',
    },
    titleWrapper: {
        paddingTop: 20,
        paddingBottom: 20,
        alignItems: "center",  
    },
    title: {
        fontSize: 30,
        fontWeight: "bold",
    },
})

export default Home;