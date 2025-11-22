import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import {
  User,
  login as apiLogin,
  register as apiRegister,
  getToken,
  getUser,
  setToken,
  setUser,
  removeToken,
  RegisterRequest,
} from "../utils/api";

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (
    email: string,
    password: string,
    displayName: string,
    userType?: "View" | "Admin" | "Analyst" | "Security"
  ) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUserState] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is already logged in
    const token = getToken();
    const userData = getUser();
    if (token && userData) {
      setUserState(userData);
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const response = await apiLogin({ email, password });
      setUserState(response.user);
    } catch (error) {
      throw error;
    }
  };

  const register = async (
    email: string,
    password: string,
    displayName: string,
    userType?: "View" | "Admin" | "Analyst" | "Security"
  ) => {
    try {
      const response = await apiRegister({
        email,
        password,
        display_name: displayName,
        user_type: userType || "View",
      } as RegisterRequest);
      setUserState(response.user);
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    removeToken();
    setUserState(null);
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
