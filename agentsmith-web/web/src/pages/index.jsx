import React from 'react';
import { withAuthenticationRequired, withAuth0 } from '@auth0/auth0-react';
import BaseComponent from "../components/BaseComponent/index.jsx";
import {Bloc} from "./bloc.js";
import {Route, Routes} from "react-router-dom";
import Shell from "../components/Shell/index.jsx";
import Dashboard from "./Dashboard/index.jsx";
import {AppContextProvider} from "./context.jsx";
import CreateProfile from "./CreateProfile/index.jsx";
class PrivateRoute extends BaseComponent {

  constructor(props) {
    super(props);
    this.state = { };
    this.setBloc(new Bloc({ auth0: this.props.auth0 }));
  }

  componentDidMount() {
    super.componentDidMount();
    this.bloc.initialise();
  }

  render() {

    const { initialised, profile, error } = this.state;

    if(!initialised) {
      return <>Loading...</>
    }

    const context = {
      bloc: this.bloc,
    };

    if(!profile) {
      return <AppContextProvider value={context}>
        <CreateProfile />
      </AppContextProvider> ;
    }

    return <AppContextProvider value={context}>
      <Shell>
        <Routes>
          <Route exact path={""} element={ <Dashboard /> }></Route>
        </Routes>
      </Shell>
    </AppContextProvider> ;
  }
}

export default withAuthenticationRequired(withAuth0(PrivateRoute), {
  onRedirecting: () => <div>Redirecting you to the login page...</div>,
});