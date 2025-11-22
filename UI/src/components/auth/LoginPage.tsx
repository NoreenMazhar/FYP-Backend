import { useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { LogoIcon } from "../dashboard/LogoIcon";
import { Loader2, Mail, Lock, Eye, EyeOff } from "lucide-react";
import { toast } from "sonner";

export function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email.trim() || !password.trim()) {
      toast.error("Please enter both email and password");
      return;
    }

    setIsLoading(true);
    try {
      await login(email.trim(), password);
      toast.success("Login successful!");
    } catch (error: any) {
      toast.error(error.message || "Invalid email or password. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4 relative overflow-hidden">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-[0.02]">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, hsl(var(--foreground)) 1px, transparent 0)`,
          backgroundSize: '24px 24px'
        }}></div>
      </div>
      <Card className="w-full max-w-md relative z-10 border-border/50 shadow-2xl bg-card/95 backdrop-blur-sm">
        <CardHeader className="space-y-6 text-center pb-8 pt-8">
          <div className="flex flex-col items-center gap-4">
            <div className="relative flex items-center justify-center">
              <div className="absolute inset-0 bg-primary/10 rounded-full blur-xl"></div>
              <div className="relative bg-primary/5 p-3 rounded-2xl border border-primary/20">
                <LogoIcon className="w-10 h-10 text-primary" />
              </div>
            </div>
            <div className="space-y-1">
              <CardTitle className="text-2xl font-bold tracking-tight">
                IntelliSecDash
              </CardTitle>
              <CardDescription className="text-sm">
                Real-time vehicle monitoring and AI-powered analytics
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email" className="flex items-center gap-2">
                <Mail className="w-4 h-4" />
                Email Address
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
                required
                className="h-11"
                autoComplete="email"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password" className="flex items-center gap-2">
                <Lock className="w-4 h-4" />
                Password
              </Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                  required
                  className="h-11 pr-10"
                  autoComplete="current-password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-0 top-0 h-11 w-10 hover:bg-transparent"
                  onClick={() => setShowPassword(!showPassword)}
                  disabled={isLoading}
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <Eye className="w-4 h-4 text-muted-foreground" />
                  )}
                </Button>
              </div>
            </div>
            <Button
              type="submit"
              className="w-full h-11 text-base font-semibold"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Signing in...
                </>
              ) : (
                "Sign In"
              )}
            </Button>
          </form>
          <div className="mt-6 pt-6 border-t border-border">
            <p className="text-xs text-center text-muted-foreground">
              Secure access to your dashboard. Your session is protected with JWT authentication.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

