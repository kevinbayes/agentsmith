import {Component} from "react";

export default class BaseComponent extends Component {

    bloc;

    constructor(props) {
        super(props);
        this.state = { initialised: false };
    }

    setBloc = (bloc) => this.bloc = bloc;

    hasInitialised = () => this.state.initialised;

    componentDidMount() {

        this.stateSubscription = this.bloc.subscribeToState(this.__handleState);
        this.eventSubscription = this.bloc.subscribeToEvents(this.__handleEvent);
    }

    componentWillUnmount() {

        if(this.stateSubscription) {
            this.stateSubscription.unsubscribe();
        }
        if(this.eventSubscription) {
            this.eventSubscription.unsubscribe();
        }
    }

    __handleState = (state) => {
        this.setState({
            ...state,
        });
    }

    __handleEvent = (event) => { }
}
